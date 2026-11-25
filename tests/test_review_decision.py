"""The decision script of the Claude review workflow: status, review event, inline threads."""

from __future__ import annotations

import importlib.util
import json
import pathlib

import pytest
from hypothesis import given
from hypothesis import strategies as st

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]

_PATH = pathlib.Path(__file__).resolve().parents[1] / '.github' / 'scripts' / 'review_decision.py'
_spec = importlib.util.spec_from_file_location('review_decision', _PATH)
rd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rd)

HEAD = 'a' * 40
BOT = 'github-actions[bot]'
POLICY_ALL = '# approve everywhere but the protected set\n{"version": 1, "approve": {"allow": ["**"], "deny": []}}\n'
PATCH = '@@ -10,3 +10,4 @@ def f():\n     a = 1\n-    b = 2\n+    b = 3\n+    c = 4\n     return a\n'


def _finding(sev='important', path='src/aragog/x.py', line=11, title='Wrong sign'):
    return {'severity': sev, 'path': path, 'line': line, 'title': title, 'body': 'b'}


def _thread(key, sev='important', author=BOT, resolved=False, marked=True, tid='T1'):
    body = 'text' + (f'\n{rd.finding_marker(key, sev)}' if marked else '')
    return {
        'id': tid,
        'isResolved': resolved,
        'path': 'src/aragog/x.py',
        'line': 11,
        'comments': {'nodes': [{'body': body, 'author': {'login': author}}]},
    }


def _plan(findings=(), verified=None, threads=(), fixed=(), fixed_ok=None, policy=POLICY_ALL,
          files=None, **kw):
    find = {'summary': 'ok', 'findings': list(findings), 'fixed': list(fixed)}
    own = rd.own_threads(list(threads))
    cands = rd.prep(find, own)
    ids = [c['id'] for c in cands['candidates']]
    verify = {
        'verdicts': [{'id': i, 'verified': True if verified is None else verified, 'reason': 'r'}
                     for i in ids],
        'fixed': [{'key': k, 'verified': True if fixed_ok is None else fixed_ok, 'reason': 'r'}
                  for k in fixed],
    }
    files = files if files is not None else [{'path': 'src/aragog/x.py', 'old_mode': '100644',
                                              'new_mode': '100644'}]
    patches = {'src/aragog/x.py': PATCH}
    return rd.plan(find, verify, cands, own, policy, files, patches, HEAD, **kw)


def test_clean_review_passes_and_approves_with_policy_all():
    p = _plan(findings=[_finding('nit')])
    assert p['state'] == 'success'
    assert p['event'] == 'APPROVE'
    assert p['comments'] == []  # a nit goes in the summary only
    assert 'Wrong sign' in p['body'] and rd.APPROVAL_MARK in p['body']
    assert p['head'] == HEAD


@pytest.mark.parametrize('sev', ['important', 'pre_existing'])
def test_blocking_finding_fails_goes_inline_and_never_approves(sev):
    p = _plan(findings=[_finding(sev)])
    assert p['state'] == 'failure'
    assert p['event'] == 'COMMENT'
    assert len(p['comments']) == 1
    c = p['comments'][0]
    assert c['line'] == 11 and c['path'] == 'src/aragog/x.py'
    assert rd.parse_marker(c['body'])[1] == sev
    assert rd.APPROVAL_MARK not in p['body']


def test_unverified_blocking_finding_is_dropped_and_counted():
    p = _plan(findings=[_finding()], verified=False)
    assert p['state'] == 'success' and p['comments'] == []
    assert '1 candidate finding was not verified' in p['body']


def test_line_outside_the_hunks_becomes_a_file_comment():
    p = _plan(findings=[_finding(line=99)])
    assert p['comments'][0].get('line') is None
    assert p['comments'][0]['subject_type'] == 'file'


def test_blocking_finding_in_a_file_outside_the_diff_stays_in_the_summary():
    p = _plan(findings=[_finding(path='src/aragog/other.py')])
    assert p['state'] == 'failure' and p['comments'] == []
    assert 'src/aragog/other.py' in p['body']


def test_thread_without_marker_is_never_resolved():
    key = rd.finding_key('src/aragog/x.py', 'Wrong sign')
    p = _plan(threads=[_thread(key, marked=False)], fixed=[key])
    assert p['resolve'] == []


def test_marked_thread_of_a_person_is_never_resolved():
    key = rd.finding_key('src/aragog/x.py', 'Wrong sign')
    p = _plan(threads=[_thread(key, author='timlichtenberg')], fixed=[key])
    assert p['resolve'] == []


def test_verified_fix_resolves_own_thread_and_passes():
    key = rd.finding_key('src/aragog/x.py', 'Wrong sign')
    p = _plan(threads=[_thread(key)], fixed=[key])
    assert p['resolve'] == ['T1']
    assert p['state'] == 'success'


def test_unverified_fix_keeps_the_thread_open_and_fails():
    key = rd.finding_key('src/aragog/x.py', 'Wrong sign')
    p = _plan(threads=[_thread(key)], fixed=[key], fixed_ok=False)
    assert p['resolve'] == [] and p['state'] == 'failure' and p['event'] == 'COMMENT'


def test_open_blocking_thread_counts_even_when_not_reported_again():
    key = rd.finding_key('src/aragog/x.py', 'Wrong sign')
    p = _plan(threads=[_thread(key, sev='pre_existing')])
    assert p['state'] == 'failure'


def test_same_finding_with_an_open_thread_gets_no_second_thread():
    key = rd.finding_key('src/aragog/x.py', 'Wrong sign')
    p = _plan(findings=[_finding()], threads=[_thread(key)])
    assert p['comments'] == [] and p['state'] == 'failure'


@pytest.mark.parametrize('path, mode', [
    ('.github/workflows/claude-review.yml', '100644'),
    ('.github/scripts/review_decision.py', '100644'),
    ('.github/review-policy.yml', '100644'),
    ('AGENTS.md', '100644'),
    ('tests/AGENTS.md', '100644'),
    ('CODEOWNERS', '100644'),
    ('docs/.gitattributes', '100644'),
    ('.gitmodules', '100644'),
    ('docs/link.md', '120000'),
    ('vendor/sub', '160000'),
])
def test_protected_change_withholds_approval(path, mode):
    files = [{'path': path, 'old_mode': '100644', 'new_mode': mode}]
    p = _plan(files=files)
    assert p['state'] == 'success'  # the status follows the findings only
    assert p['event'] == 'COMMENT'
    assert 'approval withheld' in p['body'] and path in p['body']


def test_policy_off_missing_or_broken_withholds_approval():
    off = '{"version": 1, "approve": {"allow": [], "deny": []}}'
    for policy in (off, None, '{"version": 2}', 'not json', '{"version": 1, "approve": {"allow": "**"}}'):
        assert _plan(policy=policy)['event'] == 'COMMENT', policy


def test_policy_deny_glob_wins_over_allow():
    deny = '{"version": 1, "approve": {"allow": ["**"], "deny": ["src/aragog/**"]}}'
    assert _plan(policy=deny)['event'] == 'COMMENT'
    docs = [{'path': 'docs/index.md', 'old_mode': '100644', 'new_mode': '100644'}]
    assert _plan(policy=deny, files=docs)['event'] == 'APPROVE'


def test_no_approve_flag_withholds_approval():
    assert _plan(no_approve=True)['event'] == 'COMMENT'


def test_a_failed_or_withheld_run_dismisses_own_earlier_approvals():
    assert _plan(findings=[_finding()], approvals=[7, 9])['dismiss'] == [7, 9]
    assert _plan(no_approve=True, approvals=[7])['dismiss'] == [7]
    assert _plan(approvals=[7])['dismiss'] == []


@pytest.mark.parametrize('pattern, path, hit', [
    ('**', 'a/b/c.py', True),
    ('src/aragog/**', 'src/aragog/solver/x.py', True),
    ('src/aragog/**', 'src/aragogx/x.py', False),
    ('**/CODEOWNERS', 'CODEOWNERS', True),
    ('**/CODEOWNERS', 'docs/CODEOWNERS', True),
    ('*.md', 'docs/a.md', False),
    ('docs/*.md', 'docs/a.md', True),
    ('docs/?.md', 'docs/ab.md', False),
])
def test_glob_match(pattern, path, hit):
    assert rd.glob_match(pattern, path) is hit


def test_hunk_lines_hold_added_and_context_lines_only():
    assert rd.hunk_lines(PATCH) == {10, 11, 12, 13}
    assert rd.hunk_lines(None) == set()


def test_raw_diff_parse_keeps_modes_and_both_paths_of_a_rename():
    raw = ':100644 120000 abc def T\x00docs/l.md\x00:100644 100644 abc def R100\x00a.py\x00b.py\x00'
    files = rd.parse_raw(raw)
    assert {'path': 'docs/l.md', 'old_mode': '100644', 'new_mode': '120000'} in files
    assert {f['path'] for f in files} == {'docs/l.md', 'a.py', 'b.py'}


def test_mentions_are_neutralised_outside_code():
    out = rd.neutralise('ping @tim and `@keep` and a@b.c')
    assert '@‍tim' in out and '`@keep`' in out and 'a@b.c' in out


@pytest.mark.parametrize('find', [
    None, {}, {'findings': 'x'}, {'summary': 's', 'findings': [{'severity': 'fatal', 'path': 'a',
                                                               'line': 1, 'title': 't', 'body': 'b'}]},
    {'summary': 's', 'findings': [{'severity': 'nit', 'path': 'a', 'line': True, 'title': 't',
                                   'body': 'b'}]},
])
def test_malformed_find_output_is_rejected(find):
    with pytest.raises(rd.OutputError):
        rd.prep(find, [])


@given(st.lists(st.sampled_from(['important', 'pre_existing', 'nit']), max_size=6),
       st.lists(st.booleans(), min_size=6, max_size=6))
def test_property_approval_only_when_nothing_blocking_is_verified(sevs, oks):
    find = {'summary': 's', 'findings': [_finding(s, title=f't{i}') for i, s in enumerate(sevs)]}
    cands = rd.prep(find, [])
    verify = {'verdicts': [{'id': c['id'], 'verified': ok, 'reason': ''}
                           for c, ok in zip(cands['candidates'], oks)]}
    files = [{'path': 'src/aragog/x.py', 'old_mode': '100644', 'new_mode': '100644'}]
    p = rd.plan(find, verify, cands, [], POLICY_ALL, files, {'src/aragog/x.py': PATCH}, HEAD)
    blocking = any(s in rd.BLOCKING and ok for s, ok in zip(sevs, oks))
    assert (p['state'] == 'failure') is blocking
    assert (p['event'] == 'APPROVE') is (not blocking)


class _Gh:
    """Stand-in for the gh CLI: answers the head query and records every write."""

    def __init__(self, head):
        self.head, self.calls = head, []

    def __call__(self, args, data=None):
        self.calls.append((args, data))
        return self.head + '\n' if args[:2] == ['pr', 'view'] else '{}'


def _plan_file(tmp_path, **kw):
    p = _plan(**kw)
    f = tmp_path / 'plan.json'
    f.write_text(json.dumps(p))
    return f


def test_apply_posts_nothing_when_the_head_moved(tmp_path):
    gh = _Gh('b' * 40)
    rd.apply(json.loads(_plan_file(tmp_path).read_text()), 'o/r', '5', 'url', gh)
    writes = [a for a, _ in gh.calls if a[:2] != ['pr', 'view']]
    assert len(writes) == 1 and 'statuses/' + HEAD in writes[0][-1]
    assert gh.calls[-1][1]['state'] == 'failure' and 'head moved' in gh.calls[-1][1]['description']


def test_apply_pins_the_review_to_the_head_and_sets_the_status_last(tmp_path):
    gh = _Gh(HEAD)
    rd.apply(json.loads(_plan_file(tmp_path, findings=[_finding()]).read_text()), 'o/r', '5', 'url', gh)
    paths = [a[-1] for a, _ in gh.calls if a[0] == 'api']
    assert paths[-1].endswith('statuses/' + HEAD)
    review = [d for a, d in gh.calls if a[-1].endswith('/reviews')][0]
    assert review['commit_id'] == HEAD and review['event'] == 'COMMENT'
    comment = [d for a, d in gh.calls if a[-1].endswith('/comments')][0]
    assert comment['commit_id'] == HEAD and comment['line'] == 11
