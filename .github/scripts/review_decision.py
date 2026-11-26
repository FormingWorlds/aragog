"""Decision step of the Claude review workflow (.github/workflows/claude-review.yml).

The model passes return structured findings; this script decides everything that writes to
GitHub: the commit status ``Claude review``, the review event (APPROVE or COMMENT), the inline
threads, the threads to resolve and the earlier approvals to dismiss. The workflow runs the copy
of this script from the default branch. The status is advisory: a pull_request run uses the
workflow file of the pull request, which an author with write access can change.

Subcommands
-----------
collect   the threads and approvals of this workflow -> JSON
prep      check the find output, number the candidates and write the verify input -> JSON
plan      the action plan from the find, verify and collect outputs -> JSON
apply     carry out a plan; posts nothing when the head moved since the review started. Order:
          the review with its line comments in one POST, file comments, resolved threads, the
          status, then the dismissal of earlier approvals. A failed gh call sets a failure status.

Rules
-----
- Severities: ``important`` and ``pre_existing`` block (status failure, inline thread);
  ``nit`` goes in the summary only. In a file the pull request does not change, no thread is
  possible: a ``pre_existing`` finding goes in the summary and does not block, an ``important``
  one blocks and the review body names its key; it clears when a later run no longer reports it.
  A change to a protected path (PROTECTED, a symlink, a submodule, or a path with a control
  character) also sets the status to failure. Only findings the verify pass confirmed count,
  and a verify output without exactly one verdict per candidate is an error, not a pass.
- The workflow's own threads are written by github-actions[bot] and end with
  ``<!-- claude-review:finding key=... severity=... -->``. A thread is resolved only when the
  find pass claims the fix and the verify pass confirms it; no other thread is ever resolved.
  A thread that a person other than the pull request author resolved dismisses its finding.
- APPROVE needs: no blocking finding or open blocking thread, a valid policy file whose
  ``allow`` globs match every changed path and whose ``deny`` globs match none, at least one
  changed path, and no change to a protected path.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import pathlib
import re
import subprocess
import sys

BOT = 'github-actions[bot]'  # the workflow token in REST and as a GraphQL thread resolver
BOT_GRAPHQL = 'github-actions'  # the same bot as a GraphQL comment author
CONTEXT = 'Claude review'
BLOCKING = ('important', 'pre_existing')
SEVERITIES = (*BLOCKING, 'nit')
LABEL = {'important': 'Important', 'pre_existing': 'Pre-existing', 'nit': 'Nit'}
APPROVAL_MARK = '<!-- claude-review:approval -->'
MOVED = 'head moved during the review; comment @claude review'
PROTECTED = (
    '.github/**',
    '**/AGENTS.md',
    '**/CLAUDE.md',
    '**/CLAUDE.local.md',
    '**/GEMINI.md',
    '**/.claude/**',
    '**/.mcp.json',
    '**/conftest.py',
    'pyproject.toml',
    '.pre-commit-config.yaml',
    'tools/agents/**',
    '**/CODEOWNERS',
    '**/.gitattributes',
    '**/.gitmodules',
)
_MARK = re.compile(
    r'<!-- claude-review:finding key=([0-9a-f]{12}) severity=(important|pre_existing|nit) -->\s*\Z'
)
_RESOLVE = 'mutation($t: ID!) { resolveReviewThread(input: {threadId: $t}) { thread { id } } }'
_THREADS = """query($o: String!, $r: String!, $n: Int!, $c: String) {
  repository(owner: $o, name: $r) { pullRequest(number: $n) { author { login }
    reviewThreads(first: 100, after: $c) { pageInfo { hasNextPage endCursor }
      nodes { id isResolved resolvedBy { login } path line
        comments(first: 1) { nodes { body author { login } } } } } } } }"""


class OutputError(ValueError):
    """A model pass returned output that does not match its schema."""


def finding_key(path, title):
    """Return the key that ties a finding to its thread across runs."""
    text = path + '\n' + ' '.join(title.lower().split())
    return hashlib.sha1(text.encode()).hexdigest()[:12]


def finding_marker(key, severity):
    return f'<!-- claude-review:finding key={key} severity={severity} -->'


def parse_marker(body):
    """Return (key, severity) of the finding marker that ends ``body``, or None."""
    m = _MARK.search(body or '')
    return (m[1], m[2]) if m else None


def glob_match(pattern, path):
    """Match a path glob: ``**`` spans directories, ``*`` and ``?`` stay inside one."""
    rx = ''
    for tok in re.split(r'(\*\*/|\*\*|\*|\?)', pattern):
        rx += {'**/': '(?:.*/)?', '**': '.*', '*': '[^/]*', '?': '[^/]'}.get(
            tok, re.escape(tok)
        )
    return re.fullmatch(rx, path, re.S) is not None


def hunk_lines(patch):
    """Return the new-side line numbers inside the hunks of a unified diff patch."""
    lines, n = set(), None
    for line in (patch or '').splitlines():
        if m := re.match(r'@@ -\S+ \+(\d+)', line):
            n = int(m[1])
        elif n is not None and line[:1] in ('', ' ', '+'):
            lines.add(n)
            n += 1
    return lines


def parse_raw(raw):
    """Parse ``git diff --raw -z``: one entry per path, a rename gives both paths."""
    parts, files, i = raw.split('\0'), [], 0
    while i < len(parts) and parts[i].startswith(':'):
        old_mode, new_mode, _, _, status = parts[i][1:].split()
        n = 2 if status[0] in 'RC' else 1
        files += [
            {'path': p, 'old_mode': old_mode, 'new_mode': new_mode}
            for p in parts[i + 1 : i + 1 + n]
        ]
        i += 1 + n
    if any(parts[i:]):
        raise ValueError(f'unparsed raw diff from entry {i}: {parts[i][:80]!r}')
    return files


def neutralise(text):
    """Break @mentions outside code spans and drop HTML comments, so model text notifies
    nobody and cannot carry a marker."""
    while '<!--' in text or '-->' in text:
        text = text.replace('<!--', '').replace('-->', '')
    parts = re.split(r'(```.*?```|`[^`\n]*`)', text, flags=re.S)
    return ''.join(
        p if k % 2 else re.sub(r'(?<![\w.])@(?=[A-Za-z0-9_-])', '@‍', p)
        for k, p in enumerate(parts)
    )


def load_policy(text):
    """Return the approve globs of the policy file, or None when it is missing or invalid.

    The file is YAML in flow style, so after its ``#`` comment lines it is also JSON.
    """
    try:
        body = '\n'.join(ln for ln in text.splitlines() if not ln.lstrip().startswith('#'))
        pol = json.loads(body)
        allow, deny = pol['approve']['allow'], pol['approve']['deny']
    except (AttributeError, TypeError, KeyError, ValueError):
        return None
    ok = type(pol.get('version')) is int and pol['version'] == 1
    ok = ok and all(
        isinstance(g, list) and all(isinstance(x, str) for x in g) for g in (allow, deny)
    )
    return {'allow': allow, 'deny': deny} if ok else None


def own_threads(nodes, pr_author):
    """Return the threads of this workflow: bot-authored and ending with a finding marker.

    ``state`` is ``open`` when the thread is unresolved, or resolved by the pull request author,
    by an unknown actor or while the author is unknown; ``fixed`` when the workflow resolved it;
    ``dismissed`` when another person resolved it.
    """
    out = []
    for t in nodes:
        first = (t.get('comments', {}).get('nodes') or [{}])[0]
        mark = parse_marker(first.get('body'))
        if mark and (first.get('author') or {}).get('login') == BOT_GRAPHQL:
            by = (t.get('resolvedBy') or {}).get('login')
            state = 'open'
            if t.get('isResolved') and by == BOT:
                state = 'fixed'
            elif t.get('isResolved') and by and pr_author and by != pr_author:
                state = 'dismissed'
            out.append(
                {
                    'id': t['id'],
                    'key': mark[0],
                    'severity': mark[1],
                    'state': state,
                    'resolved': bool(t.get('isResolved')),
                    'path': t.get('path'),
                    'line': t.get('line'),
                    'text': _MARK.sub('', first['body']).strip()[:1500],
                }
            )
    return out


def prep(find, threads):
    """Check the find output, number its findings and write the verify input.

    A finding with an open or dismissed thread of this workflow is not a candidate; a finding
    whose thread the workflow resolved as fixed is, so a returning bug blocks again.
    """
    if (
        not isinstance(find, dict)
        or not isinstance(find.get('summary'), str)
        or not isinstance(find.get('findings'), list)
        or not isinstance(find.get('fixed'), list)
        or not all(isinstance(k, str) for k in find['fixed'])
    ):
        raise OutputError('the find pass returned no valid structured output')
    seen, cands = {t['key'] for t in threads if t['state'] != 'fixed'}, []
    for f in find['findings']:
        ok = (
            isinstance(f, dict)
            and f.get('severity') in SEVERITIES
            and type(f.get('line')) is int
            and all(isinstance(f.get(k), str) for k in ('path', 'title', 'body'))
        )
        if not ok:
            raise OutputError(f'the find pass returned a malformed finding: {str(f)[:200]}')
        f = {**f, 'path': f['path'].removeprefix('./')}
        key = finding_key(f['path'], f['title'])
        if key not in seen:
            seen.add(key)
            cands.append({**f, 'key': key, 'id': f'c{len(cands) + 1}'})
    open_keys = {t['key'] for t in threads if t['state'] == 'open'}
    fixed = [k for k in dict.fromkeys(find['fixed']) if k in open_keys]
    shown = ('id', 'severity', 'path', 'line', 'title', 'body')
    return {
        'candidates': cands,
        'threads': threads,
        'verify_in': {'candidates': [{k: c[k] for k in shown} for c in cands], 'fixed': fixed},
    }


def _verdicts(verify, cands):
    """Return the candidate ids and fixed keys the verify pass confirmed; fail on a gap."""
    if (
        not isinstance(verify, dict)
        or not isinstance(verify.get('verdicts'), list)
        or not isinstance(verify.get('fixed'), list)
    ):
        raise OutputError('the verify pass returned no valid structured output')
    rows = verify['verdicts'] + verify['fixed']
    if not all(isinstance(v, dict) and type(v.get('verified')) is bool for v in rows):
        raise OutputError('the verify pass returned a malformed verdict')
    ids = [v.get('id') for v in verify['verdicts']]
    want = [c['id'] for c in cands['candidates']]
    if sorted(map(str, ids)) != sorted(want):
        raise OutputError(
            f'the verify pass gave verdicts for {ids}, expected one for each of {want}'
        )
    return (
        {v['id'] for v in verify['verdicts'] if v['verified']},
        {v.get('key') for v in verify['fixed'] if v['verified']},
    )


def _protected(files):
    return [
        f['path']
        for f in files
        if {f['old_mode'], f['new_mode']} & {'120000', '160000'}
        or re.search(r'[\x00-\x1f\x7f]', f['path'])
        or any(glob_match(g, f['path']) for g in PROTECTED)
    ]


def _withheld(policy_text, files):
    pol = load_policy(policy_text)
    if pol is None:
        return ['no valid .github/review-policy.yml on the default branch']
    if not files:
        return ['the pull request changes no file']
    return [
        f'the policy does not allow approval for {f["path"]}'
        for f in files
        if not any(glob_match(g, f['path']) for g in pol['allow'])
        or any(glob_match(g, f['path']) for g in pol['deny'])
    ]


def plan(
    find,
    verify,
    cands,
    threads,
    policy_text,
    files,
    patches,
    head,
    approvals=(),
):
    """Return the action plan for one review run (see the module docstring)."""
    ids, fixed_ok = _verdicts(verify, cands)
    fixed = fixed_ok & set(cands['verify_in']['fixed'])
    found = [c for c in cands['candidates'] if c['id'] in ids]
    rejected = len(cands['candidates']) - len(found)
    open_threads = [t for t in threads if t['state'] == 'open']
    still_open = [
        t for t in open_threads if t['key'] not in fixed and t['severity'] in BLOCKING
    ]
    blocking = [c for c in found if c['severity'] in BLOCKING]
    changed = {f['path'] for f in files}

    comments, outside, unthreaded = [], [], []
    for c in blocking:
        body = (
            neutralise(f'**{LABEL[c["severity"]]}**: {c["title"]}\n\n{c["body"]}')[:59000]
            + '\n\n'
            + finding_marker(c['key'], c['severity'])
        )
        if c['line'] in hunk_lines(patches.get(c['path'])):
            comments.append(
                {'path': c['path'], 'line': c['line'], 'side': 'RIGHT', 'body': body}
            )
        elif c['path'] in changed:
            comments.append({'path': c['path'], 'subject_type': 'file', 'body': body})
        elif c['severity'] == 'pre_existing':
            outside.append(c)
        else:
            unthreaded.append(c)

    n_block = len(comments) + len(unthreaded) + len(still_open)
    prot = _protected(files)
    why = n_block or prot or _withheld(policy_text, files)
    event = 'COMMENT' if why else 'APPROVE'

    counts = {s: sum(c['severity'] == s for c in found) for s in SEVERITIES}
    lines = [APPROVAL_MARK] if event == 'APPROVE' else []
    lines += [
        neutralise(find['summary']),
        '',
        '| Important | Pre-existing | Nit |',
        '|---|---|---|',
        f'| {counts["important"]} | {counts["pre_existing"]} | {counts["nit"]} |',
    ]
    nits = [c for c in found if c['severity'] == 'nit'] + outside
    if nits:
        lines += ['', '**Findings without a thread**'] + [
            neutralise(
                f'- {LABEL[c["severity"]]}: `{c["path"]}:{c["line"]}` {c["title"]}: {c["body"]}'
            )
            for c in nits
        ]
    if unthreaded:
        lines += ['', '**Blocking findings in files this pull request does not change**'] + [
            neutralise(
                f'- `{c["path"]}:{c["line"]}` {c["title"]}: {c["body"]} (key {c["key"]})'
            )
            for c in unthreaded
        ]
    if still_open:
        lines += ['', f'{len(still_open)} earlier blocking thread(s) stay open.']
    if rejected:
        lines += [
            '',
            f'{rejected} candidate finding{" was" if rejected == 1 else "s were"} not verified and left out.',
        ]
    if prot:
        lines += [
            '',
            'A maintainer has to review the protected paths: '
            + ', '.join(map(repr, prot))
            + '.',
        ]
    if event == 'APPROVE':
        lines += ['', 'Approved by the review workflow. A code owner still has to approve.']
    elif not n_block and not prot:
        lines += ['', 'The status passes; approval withheld: ' + '; '.join(why) + '.']
    if n_block:
        desc = f'{n_block} blocking finding(s)'
    else:
        desc = 'protected path changed: maintainer review' if prot else 'no blocking findings'
    return {
        'head': head,
        'state': 'failure' if n_block or prot else 'success',
        'description': desc,
        'event': event,
        'body': '\n'.join(lines),
        'comments': comments,
        'resolve': sorted(
            t['id'] for t in open_threads if t['key'] in fixed and not t['resolved']
        ),
        'dismiss': [] if event == 'APPROVE' else list(approvals),
    }


def gh(args, data=None):
    """Run the gh CLI; ``data`` goes to ``gh api --input -`` as JSON."""
    cmd = ['gh', *args] + (['--input', '-'] if data is not None else [])
    res = subprocess.run(
        cmd,
        input=json.dumps(data) if data is not None else None,
        capture_output=True,
        text=True,
        check=True,
    )
    return res.stdout


def collect(repo, pr, run=gh):
    """Return the threads of this workflow (all pages) and the ids of its earlier approvals."""
    owner, name = repo.split('/')
    nodes, cursor, author = [], None, None
    while True:
        args = [
            'api',
            'graphql',
            '-f',
            f'query={_THREADS}',
            '-f',
            f'o={owner}',
            '-f',
            f'r={name}',
            '-F',
            f'n={pr}',
        ]
        page = json.loads(run(args + (['-f', f'c={cursor}'] if cursor else [])))
        pull = page['data']['repository']['pullRequest']
        author, threads = (pull.get('author') or {}).get('login'), pull['reviewThreads']
        nodes += threads['nodes']
        if not threads['pageInfo']['hasNextPage']:
            break
        cursor = threads['pageInfo']['endCursor']
    reviews = json.loads(
        run(['api', '--paginate', '--slurp', f'repos/{repo}/pulls/{pr}/reviews'])
    )
    approvals = [
        r['id']
        for page in reviews
        for r in page
        if r['state'] == 'APPROVED'
        and (r.get('user') or {}).get('login') == BOT
        and APPROVAL_MARK in (r.get('body') or '')
    ]
    return {'threads': own_threads(nodes, author), 'approvals': approvals}


def set_status(repo, sha, state, description, url, run=gh):
    data = {
        'state': state,
        'context': CONTEXT,
        'description': description[:140],
        'target_url': url,
    }
    run(['api', '-X', 'POST', f'repos/{repo}/statuses/{sha}'], data)


def _head(repo, pr, run):
    return run(
        ['pr', 'view', pr, '--repo', repo, '--json', 'headRefOid', '--jq', '.headRefOid']
    ).strip()


def apply(p, repo, pr, url, run=gh):
    """Carry out a plan; posts nothing but a failure status when the head moved.

    An approval is dismissed again when the head moved while it was posted. A failed gh call
    sets a failure status with its error and is raised again.
    """
    if _head(repo, pr, run) != p['head']:
        return set_status(repo, p['head'], 'failure', MOVED, url, run)
    try:
        _post(p, repo, pr, url, run)
    except subprocess.CalledProcessError as err:
        cause = (err.stderr or '').strip().splitlines()[-1:] or [f'exit {err.returncode}']
        with contextlib.suppress(subprocess.CalledProcessError):
            set_status(repo, p['head'], 'failure', f'post failed: {cause[0]}', url, run)
        raise


def _post(p, repo, pr, url, run):
    inline = [c for c in p['comments'] if 'line' in c]
    review = run(
        ['api', '-X', 'POST', f'repos/{repo}/pulls/{pr}/reviews'],
        {
            'commit_id': p['head'],
            'event': p['event'],
            'body': p['body'][:60000],
            'comments': inline,
        },
    )
    for c in p['comments']:
        if 'line' not in c:
            run(
                ['api', '-X', 'POST', f'repos/{repo}/pulls/{pr}/comments'],
                {**c, 'commit_id': p['head']},
            )
    for tid in p['resolve']:
        run(['api', 'graphql', '-f', f'query={_RESOLVE}', '-F', f't={tid}'])
    if p['event'] == 'APPROVE' and _head(repo, pr, run) != p['head']:
        run(
            [
                'api',
                '-X',
                'PUT',
                f'repos/{repo}/pulls/{pr}/reviews/{json.loads(review)["id"]}/dismissals',
            ],
            {'message': 'The pull request changed while it was approved.', 'event': 'DISMISS'},
        )
        return set_status(repo, p['head'], 'failure', MOVED, url, run)
    set_status(repo, p['head'], p['state'], p['description'], url, run)
    for rid in p['dismiss']:
        msg = 'A later review run did not approve this pull request.'
        run(
            ['api', '-X', 'PUT', f'repos/{repo}/pulls/{pr}/reviews/{rid}/dismissals'],
            {'message': msg, 'event': 'DISMISS'},
        )


def _read(path):
    return json.loads(pathlib.Path(path).read_text(encoding='utf-8'))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest='cmd', required=True)
    args = {
        'collect': ('--repo', '--pr', '--out'),
        'prep': ('--find', '--collect', '--out'),
        'plan': '--find --verify --candidates --collect --raw --files --head --out'.split(),
        'apply': ('--plan', '--repo', '--pr', '--url'),
    }
    for cmd, names in args.items():
        c = sub.add_parser(cmd)
        for name in names:
            c.add_argument(name, required=True)
    sub.choices['plan'].add_argument('--policy')
    a = ap.parse_args(argv)

    try:
        if a.cmd == 'apply':
            apply(_read(a.plan), a.repo, a.pr, a.url, gh)
            return 0
        if a.cmd == 'collect':
            out = collect(a.repo, a.pr, gh)
        elif a.cmd == 'prep':
            out = prep(_read(a.find), _read(a.collect)['threads'])
        else:
            col = _read(a.collect)
            files = parse_raw(pathlib.Path(a.raw).read_text(encoding='utf-8'))
            patches = {f['filename']: f.get('patch') for page in _read(a.files) for f in page}
            policy = None
            if a.policy:
                with contextlib.suppress(OSError):
                    policy = pathlib.Path(a.policy).read_text(encoding='utf-8')
            out = plan(
                _read(a.find),
                _read(a.verify),
                _read(a.candidates),
                col['threads'],
                policy,
                files,
                patches,
                a.head,
                col['approvals'],
            )
    except (OutputError, ValueError, subprocess.CalledProcessError) as err:
        print(f'review_decision: {err}', file=sys.stderr)
        return 2
    pathlib.Path(a.out).write_text(json.dumps(out), encoding='utf-8')
    return 0


if __name__ == '__main__':
    sys.exit(main())
