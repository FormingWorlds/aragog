"""Decision step of the Claude review workflow (.github/workflows/claude-review.yml).

The model passes return structured findings; this script decides everything that writes to
GitHub: the commit status ``Claude review``, the review event (APPROVE or COMMENT), the inline
threads, the threads to resolve and the earlier approvals to dismiss. The workflow runs the copy
of this script from the base branch, so a pull request cannot change how it is judged.

Subcommands
-----------
collect   open threads and approvals of the workflow -> JSON
prep      check the find output and number the candidates -> JSON
plan      the action plan from the find, verify and collect outputs -> JSON
apply     carry out a plan; posts nothing when the head moved since the review started

Rules
-----
- Severities: ``important`` and ``pre_existing`` block (status failure, inline thread);
  ``nit`` goes in the summary only. Only findings the verify pass confirmed count.
- The workflow's own threads carry ``<!-- claude-review:finding key=... -->`` in the first
  comment and are written by github-actions[bot]; no other thread is ever resolved.
- APPROVE needs: no blocking finding or open blocking thread, a valid policy file whose
  ``allow`` globs match every changed path and whose ``deny`` globs match none, and no change
  to a protected path (PROTECTED, a symlink or a submodule).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys

BOT = 'github-actions[bot]'
CONTEXT = 'Claude review'
BLOCKING = ('important', 'pre_existing')
SEVERITIES = (*BLOCKING, 'nit')
LABEL = {'important': 'Important', 'pre_existing': 'Pre-existing', 'nit': 'Nit'}
APPROVAL_MARK = '<!-- claude-review:approval -->'
PROTECTED = (
    '.github/**',
    'AGENTS.md',
    'tests/AGENTS.md',
    '**/CODEOWNERS',
    '**/.gitattributes',
    '**/.gitmodules',
)
_MARK = re.compile(r'<!-- claude-review:finding key=([0-9a-f]{12}) severity=([a-z_]+) -->')


class OutputError(ValueError):
    """A model pass returned output that does not match its schema."""


def finding_key(path, title):
    """Return the key that ties a finding to its thread across runs."""
    text = path + '\n' + ' '.join(title.lower().split())
    return hashlib.sha1(text.encode()).hexdigest()[:12]


def finding_marker(key, severity):
    return f'<!-- claude-review:finding key={key} severity={severity} -->'


def parse_marker(body):
    """Return (key, severity) of a finding marker in ``body``, or None."""
    m = _MARK.search(body or '')
    return (m[1], m[2]) if m else None


def glob_match(pattern, path):
    """Match a path glob: ``**`` spans directories, ``*`` and ``?`` stay inside one."""
    rx = ''
    for tok in re.split(r'(\*\*/|\*\*|\*|\?)', pattern):
        rx += {'**/': '(?:.*/)?', '**': '.*', '*': '[^/]*', '?': '[^/]'}.get(
            tok, re.escape(tok)
        )
    return re.fullmatch(rx, path) is not None


def hunk_lines(patch):
    """Return the new-side line numbers inside the hunks of a unified diff patch."""
    lines, n = set(), None
    for line in (patch or '').splitlines():
        if m := re.match(r'@@ -\S+ \+(\d+)', line):
            n = int(m[1])
        elif n is not None and line[:1] in (' ', '+'):
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
    return files


def neutralise(text):
    """Break @mentions outside code spans so a summary notifies nobody."""
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
    ok = pol.get('version') == 1 and all(
        isinstance(g, list) and all(isinstance(x, str) for x in g) for g in (allow, deny)
    )
    return {'allow': allow, 'deny': deny} if ok else None


def own_threads(nodes):
    """Return the open threads of this workflow: bot-authored and carrying a finding marker."""
    out = []
    for t in nodes:
        first = (t.get('comments', {}).get('nodes') or [{}])[0]
        mark = parse_marker(first.get('body'))
        if not t.get('isResolved') and mark and (first.get('author') or {}).get('login') == BOT:
            out.append(
                {
                    'id': t['id'],
                    'key': mark[0],
                    'severity': mark[1],
                    'path': t.get('path'),
                    'line': t.get('line'),
                }
            )
    return out


def prep(find, threads):
    """Check the find output and number its findings; drop those that already have a thread."""
    if (
        not isinstance(find, dict)
        or not isinstance(find.get('summary'), str)
        or not isinstance(find.get('findings'), list)
    ):
        raise OutputError('the find pass returned no valid structured output')
    open_keys, cands = {t['key'] for t in threads}, []
    for f in find['findings']:
        ok = (
            isinstance(f, dict)
            and f.get('severity') in SEVERITIES
            and isinstance(f.get('line'), int)
            and not isinstance(f.get('line'), bool)
            and all(isinstance(f.get(k), str) for k in ('path', 'title', 'body'))
        )
        if not ok:
            raise OutputError(f'the find pass returned a malformed finding: {str(f)[:200]}')
        key = finding_key(f['path'], f['title'])
        if key not in open_keys:
            cands.append({**f, 'key': key, 'id': f'c{len(cands) + 1}'})
    return {'candidates': cands, 'threads': threads}


def _verified(verify, field, name):
    return {
        v[name]
        for v in (verify or {}).get(field) or []
        if isinstance(v, dict) and v.get('verified') is True and name in v
    }


def _withheld(policy_text, files, no_approve):
    if no_approve:
        return ['the decision script is not on the base branch yet']
    pol = load_policy(policy_text) if policy_text is not None else None
    if pol is None:
        return ['no valid .github/review-policy.yml on the base branch']
    why = []
    for f in files:
        p = f['path']
        if any(glob_match(g, p) for g in PROTECTED) or {f['old_mode'], f['new_mode']} & {
            '120000',
            '160000',
        }:
            why.append(f'touches the protected path {p}')
        elif not any(glob_match(g, p) for g in pol['allow']) or any(
            glob_match(g, p) for g in pol['deny']
        ):
            why.append(f'the policy does not allow approval for {p}')
    return why


def plan(
    find,
    verify,
    cands,
    threads,
    policy_text,
    files,
    patches,
    head,
    no_approve=False,
    approvals=(),
):
    """Return the action plan for one review run (see the module docstring)."""
    ids = _verified(verify, 'verdicts', 'id')
    fixed = _verified(verify, 'fixed', 'key')
    found = [c for c in cands['candidates'] if c['id'] in ids]
    rejected = len(cands['candidates']) - len(found)
    still_open = [t for t in threads if t['key'] not in fixed and t['severity'] in BLOCKING]
    blocking = [c for c in found if c['severity'] in BLOCKING]
    changed = {f['path'] for f in files}

    comments, outside = [], []
    for c in blocking:
        body = neutralise(
            f'**{LABEL[c["severity"]]}**: {c["title"]}\n\n{c["body"]}\n\n'
            f'{finding_marker(c["key"], c["severity"])}'
        )
        if c['line'] in hunk_lines(patches.get(c['path'])):
            comments.append(
                {'path': c['path'], 'line': c['line'], 'side': 'RIGHT', 'body': body}
            )
        elif c['path'] in changed:
            comments.append({'path': c['path'], 'subject_type': 'file', 'body': body})
        else:
            outside.append(c)

    n_block = len(blocking) + len(still_open)
    state = 'failure' if n_block else 'success'
    why = (
        _withheld(policy_text, files, no_approve)
        if state == 'success'
        else ['blocking findings']
    )
    event = 'APPROVE' if not why else 'COMMENT'

    counts = {s: sum(c['severity'] == s for c in found) for s in SEVERITIES}
    lines = [
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
    if still_open:
        lines += ['', f'{len(still_open)} earlier blocking thread(s) stay open.']
    if rejected:
        lines += [
            '',
            f'{rejected} candidate finding{" was" if rejected == 1 else "s were"} not verified '
            'and left out.',
        ]
    if event == 'APPROVE':
        lines += [
            '',
            'Approved by the review workflow. A code owner still has to approve.',
            APPROVAL_MARK,
        ]
    elif state == 'success':
        lines += ['', 'The status passes; approval withheld: ' + '; '.join(why) + '.']
    desc = f'{n_block} blocking finding(s)' if n_block else 'no blocking findings'
    return {
        'head': head,
        'state': state,
        'description': desc,
        'event': event,
        'body': '\n'.join(lines),
        'comments': comments,
        'resolve': sorted(t['id'] for t in threads if t['key'] in fixed),
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


_THREADS = """query($o: String!, $r: String!, $n: Int!) { repository(owner: $o, name: $r) {
  pullRequest(number: $n) { reviewThreads(first: 100) { nodes { id isResolved path line
    comments(first: 1) { nodes { body author { login } } } } } } } }"""


def collect(repo, pr, run=gh):
    """Return the open own threads and the ids of the earlier own approvals."""
    owner, name = repo.split('/')
    out = json.loads(
        run(
            [
                'api',
                'graphql',
                '-f',
                f'query={_THREADS}',
                '-F',
                f'o={owner}',
                '-F',
                f'r={name}',
                '-F',
                f'n={pr}',
            ]
        )
    )
    nodes = out['data']['repository']['pullRequest']['reviewThreads']['nodes']
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
    return {'threads': own_threads(nodes), 'approvals': approvals}


def set_status(repo, sha, state, description, url, run=gh):
    run(
        ['api', '-X', 'POST', f'repos/{repo}/statuses/{sha}'],
        {
            'state': state,
            'context': CONTEXT,
            'description': description[:140],
            'target_url': url,
        },
    )


def apply(p, repo, pr, url, run=gh):
    """Carry out a plan; posts nothing but a failure status when the head moved."""
    head = run(
        ['pr', 'view', pr, '--repo', repo, '--json', 'headRefOid', '--jq', '.headRefOid']
    ).strip()
    if head != p['head']:
        set_status(
            repo,
            p['head'],
            'failure',
            'head moved during the review; comment @claude review',
            url,
            run,
        )
        return
    for c in p['comments']:
        run(
            ['api', '-X', 'POST', f'repos/{repo}/pulls/{pr}/comments'],
            {**c, 'commit_id': p['head']},
        )
    for tid in p['resolve']:
        run(
            [
                'api',
                'graphql',
                '-f',
                'query=mutation($t: ID!) { resolveReviewThread(input: {threadId: $t}) { thread { id } } }',
                '-F',
                f't={tid}',
            ]
        )
    for rid in p['dismiss']:
        run(
            ['api', '-X', 'PUT', f'repos/{repo}/pulls/{pr}/reviews/{rid}/dismissals'],
            {
                'message': 'A later review run did not approve this pull request.',
                'event': 'DISMISS',
            },
        )
    run(
        ['api', '-X', 'POST', f'repos/{repo}/pulls/{pr}/reviews'],
        {'commit_id': p['head'], 'event': p['event'], 'body': p['body'][:60000]},
    )
    set_status(repo, p['head'], p['state'], p['description'], url, run)


def _read(path):
    with open(path, encoding='utf-8') as fh:
        return json.load(fh)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest='cmd', required=True)
    c = sub.add_parser('collect')
    c.add_argument('--repo', required=True)
    c.add_argument('--pr', required=True)
    c = sub.add_parser('prep')
    c.add_argument('--find', required=True)
    c.add_argument('--collect', required=True)
    c = sub.add_parser('plan')
    for a in ('--find', '--verify', '--candidates', '--collect', '--raw', '--files', '--head'):
        c.add_argument(a, required=True)
    c.add_argument('--policy')
    c.add_argument('--no-approve', action='store_true')
    c = sub.add_parser('apply')
    for a in ('--plan', '--repo', '--pr', '--url'):
        c.add_argument(a, required=True)
    a = ap.parse_args(argv)

    if a.cmd == 'collect':
        out = collect(a.repo, a.pr)
    elif a.cmd == 'prep':
        out = prep(_read(a.find), _read(a.collect)['threads'])
    elif a.cmd == 'plan':
        col, cands = _read(a.collect), _read(a.candidates)
        with open(a.raw, encoding='utf-8') as fh:
            files = parse_raw(fh.read())
        patches = {f['filename']: f.get('patch') for page in _read(a.files) for f in page}
        policy = None
        if a.policy:
            try:
                with open(a.policy, encoding='utf-8') as fh:
                    policy = fh.read()
            except OSError:
                policy = None
        out = plan(
            _read(a.find),
            _read(a.verify),
            cands,
            col['threads'],
            policy,
            files,
            patches,
            a.head,
            a.no_approve,
            col['approvals'],
        )
    else:
        apply(_read(a.plan), a.repo, a.pr, a.url)
        return 0
    json.dump(out, sys.stdout)
    return 0


if __name__ == '__main__':
    sys.exit(main())
