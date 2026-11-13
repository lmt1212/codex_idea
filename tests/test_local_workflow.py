import unittest
from unittest.mock import patch
import sys
from codes.multihop_qa.extraction import extract
from codes.multihop_qa.demo import fixture_documents,FixtureClient,fixture_facts
from codes.multihop_qa.ingestion import chunk_documents
from codes.multihop_qa.generation import generate_qa
from codes.multihop_qa.extraction import validate_fact
from codes.multihop_qa.retrieval import CandidateIndex
from codes.multihop_qa.mining import mine_paths


class StagedClient:
    extraction_strategy='staged'
    def __init__(self,subject='企业甲'):
        self.subject=subject
    def ask(self,task,instruction,data):
        if task=='select_source':return {'quote':'企业甲采用知识图谱。'}
        if task=='extract_triple':return dict(subject=self.subject,predicate='采用',object='知识图谱')
        return dict(role='adoption',conditions=[],time='',polarity='affirmed',modality='certain',event='')


class LocalWorkflowTests(unittest.TestCase):
    def test_known_definition_fragment_cannot_pass_model_self_review(self):
        chunk=chunk_documents(fixture_documents())[0]
        chunk['text']='为一种组织致密的稀土金属，通常是由海绵稀土金属制得。'
        client=StagedClient();client.require_fact_review=True
        def ask(task,instruction,data):
            if task=='select_source':return dict(quote=chunk['text'])
            if task=='extract_triple':return dict(subject='为一种组织致密的稀土金属',predicate='通常是由',object='海绵稀土金属')
            self.fail('fragment must be rejected before model self review')
        client.ask=ask
        with self.assertRaisesRegex(ValueError,'definition fragment'):extract(chunk,client)

    def test_missing_qualifier_overrules_conflicting_positive_boolean(self):
        from codes.multihop_qa.verification import verify_support
        client=FixtureClient('judge')
        result=dict(claims=[dict(claim_id='c',entailed=True,qualifiers_complete=True,citations_correct=True,missing_qualifiers=['仅高温'])],question_clear=True,answer_complete=True,unique_answer=True,connected_reasoning=True)
        with patch.object(client,'ask',return_value=result):
            check=verify_support(dict(answer_claims=[dict(claim_id='c')],evidence_chunk_ids=['e']),{'e':dict(text='仅高温')},client)
        self.assertFalse(check['passed'])
        self.assertEqual(len(check['diagnostic_issues']),1)

    def test_unsupported_qualifier_rejection_stays_rejected_and_is_flagged(self):
        from codes.multihop_qa.verification import verify_support
        client=FixtureClient('judge')
        result=dict(claims=[dict(claim_id='c',entailed=True,qualifiers_complete=False,citations_correct=True)],question_clear=True,answer_complete=True,unique_answer=True,connected_reasoning=True)
        with patch.object(client,'ask',return_value=result):
            check=verify_support(dict(answer_claims=[dict(claim_id='c')],evidence_chunk_ids=['e']),{'e':dict(text='原文')},client)
        self.assertFalse(check['passed'])
        self.assertEqual(len(check['diagnostic_issues']),1)

    def test_semantic_completeness_gate_rejects_anchored_but_incomplete_fact(self):
        client=StagedClient()
        client.require_fact_review=True
        original=client.ask
        def ask(task,instruction,data):
            if task=='fact_completeness':return dict(subject_explicit=False,relation_complete=True,qualifiers_complete=True,reason='fragment')
            return original(task,instruction,data)
        client.ask=ask
        with self.assertRaisesRegex(ValueError,'completeness rejected'):
            extract(chunk_documents(fixture_documents())[0],client)

    def test_failed_local_call_preserves_complete_response_and_budget(self):
        from codes.multihop_qa.providers import LocalJSONClient,parse_model_object
        client=LocalJSONClient('unused',max_calls=1)
        raw='x'*2000+'[]}[]'
        def broken(task,instruction,data,record):
            record['raw_response']=raw
            return parse_model_object(raw)
        with patch.object(client,'_ask',side_effect=broken):
            with self.assertRaises(ValueError):client.ask('generation','instruction',{'a':1})
            with self.assertRaisesRegex(ValueError,'budget'):client.ask('generation','instruction',{})
        self.assertEqual(len(client.history),1)
        self.assertEqual(client.history[0]['raw_response'],raw)
        self.assertEqual(client.history[0]['status'],'failed')
        self.assertGreaterEqual(client.history[0]['elapsed_seconds'],0)

    def test_model_load_failure_is_audited_without_response(self):
        from codes.multihop_qa.providers import LocalJSONClient
        client=LocalJSONClient('unused')
        with patch.object(client,'_ask',side_effect=RuntimeError('load failed')):
            with self.assertRaises(RuntimeError):client.ask('select_source','prompt',{})
        self.assertIsNone(client.history[0]['raw_response'])
        self.assertEqual(client.history[0]['error_type'],'RuntimeError')

    def test_local_success_has_reproducible_request_fingerprint(self):
        from codes.multihop_qa.providers import LocalJSONClient
        client=LocalJSONClient('unused')
        with patch.object(client,'_ask',return_value={'ok':True}):
            client.ask('test','prompt',{'b':2,'a':1})
            client.ask('test','prompt',{'a':1,'b':2})
        self.assertEqual(client.history[0]['request_sha256'],client.history[1]['request_sha256'])
        self.assertEqual(client.history[0]['status'],'parsed')

    def test_generate_cli_uses_local_backends_by_default(self):
        from codes.multihop_qa import __main__ as cli
        generator,verifier=FixtureClient('gen'),FixtureClient('judge')
        argv=['epc','generate','input','--output','unused','--live','--generator-model','gen','--verifier-model','judge']
        with patch.object(sys,'argv',argv),patch.object(cli,'LocalJSONClient',side_effect=[generator,verifier]) as local,patch.object(cli,'JSONClient') as api,patch.object(cli,'generation_stage',return_value={}) as stage:
            self.assertEqual(cli.main(),0)
        self.assertEqual(local.call_count,2)
        api.assert_not_called()
        self.assertIs(stage.call_args.args[2],generator)
        self.assertIs(stage.call_args.args[3],verifier)

    def test_staged_extraction_preserves_source(self):
        chunk=chunk_documents(fixture_documents())[0]
        fact=extract(chunk,StagedClient())[0]
        self.assertEqual(chunk['text'][fact['start']:fact['end']],fact['quote'])
        self.assertEqual(fact['status'],'quote_validated_pending_semantic_review')

    def test_staged_extraction_rejects_changed_entity(self):
        chunk=chunk_documents(fixture_documents())[0]
        with self.assertRaises(ValueError):extract(chunk,StagedClient('虚构企业'))

    def test_same_model_review_cannot_be_mistaken_for_independent(self):
        chunks=chunk_documents(fixture_documents())
        facts=[validate_fact(f,c) for f,c in zip(fixture_facts(chunks),chunks)]
        index=CandidateIndex(facts)
        path=mine_paths(facts,[e for f in facts for e in index.recall(f)])[0]
        mapping={f['fact_id']:f for f in facts}
        evidence={c['chunk_id']:c for c in chunks}
        client=FixtureClient('same')
        with self.assertRaises(ValueError):generate_qa(path,mapping,evidence,client,client)
        qa,_=generate_qa(path,mapping,evidence,client,client,review_mode='same-model-diagnostic')
        self.assertEqual(qa['verification_status'],'diagnostic_pending_independent_and_human_review')
        self.assertFalse(qa['independent_model_review'])
