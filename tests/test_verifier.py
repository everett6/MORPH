import pytest
from verifier import LeanClient


def test_verifier_rejects_trust_bypasses_and_missing_environment(tmp_path):
    client=LeanClient(log_path=tmp_path/'results.jsonl')
    assert not client.verify('True := by trivial\nexample : True','by trivial').success
    for proof in ['by sorry','by admit','by exact (by sorry)','by\n axiom bad : False']:
        result=client.verify('False',proof)
        assert not result.success and result.proof_hash is None
    assert (tmp_path/'results.jsonl').read_text().count('\n')==5
    if not client.available:
        assert not client.verify('True','by trivial').success


@pytest.mark.skipif(not LeanClient().available,reason='real Lean 4 not installed')
def test_real_lean_exact_statement_and_invalid_proof():
    client=LeanClient(timeout=10)
    assert client.verify('True -> True','by intro h; exact h').success
    assert not client.verify('False','by trivial').success


def test_axiom_audit_fails_closed_on_unknown_or_duplicate_format():
    from verifier.lean_client import accepted_axiom_audit
    assert accepted_axiom_audit("'morph_candidate' does not depend on any axioms\n")
    assert accepted_axiom_audit("'morph_candidate' depends on axioms: [propext, Classical.choice]\n")
    assert not accepted_axiom_audit('morph_candidate axioms unknown format')
    assert not accepted_axiom_audit("'morph_candidate' depends on axioms: [sorryAx]\n")
    assert not accepted_axiom_audit("'morph_candidate' does not depend on any axioms\n"*2)
