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
    def test_format_retry_is_bounded_and_preserves_both_attempts(self):
        from codes.multihop_qa.providers import LocalJSONClient,parse_model_object
        client=LocalJSONClient('unused',max_calls=8)
        def invalid(task,instruction,data,record):
            record['raw_response']='{}"'
            return parse_model_object(record['raw_response'])
        with patch.object(client,'_ask',side_effect=invalid):
            with self.assertRaises(ValueError):client.ask('coverage','prompt',{})
        self.assertEqual(len(client.history),2)
        self.assertEqual(client.history[1]['retry_of'],0)
        self.assertTrue(all(c['status']=='failed' for c in client.history))

    def test_catalogue_citations_and_empty_control_are_evidence_scoped(self):
        from codes.multihop_qa.verification import verify_necessity
        chunks=chunk_documents(fixture_documents())[:2]
        qa=dict(question='组合问题',evidence_chunk_ids=[c['chunk_id'] for c in chunks],
                answer_claims=[dict(claim_id=str(i),text=c['text'],evidence_chunk_ids=['hidden']) for i,c in enumerate(chunks)])
        client=FixtureClient('judge');original=client.ask
        def ask(task,instruction,data):
            if task=='answer_coverage':
                self.assertTrue(data['evidence'])
                self.assertTrue(all(set(c)=={'claim_id','text'} for c in data['target_claims']))
                result=original(task,instruction,data)
                for row in result['claims']:
                    row['citations']=[entry['citation_id'] for entry in data['allowed_citations']
                                      if any(c['chunk_id']==entry['chunk_id'] for c in row['citations'])]
                return result
            return original(task,instruction,data)
        client.ask=ask
        result=verify_necessity(qa,{c['chunk_id']:c for c in chunks},client)
        self.assertTrue(result['passed'])
        self.assertEqual(len(result['trials']),4)
        self.assertEqual(result['trials'][0]['score'],0)
        self.assertTrue(any(t['claims'][0]['citations'] for t in result['trials']))

    def test_unknown_catalogue_citation_is_an_error_not_zero_coverage(self):
        from codes.multihop_qa.verification import verify_necessity
        chunks=chunk_documents(fixture_documents())[:2]
        qa=dict(question='组合问题',evidence_chunk_ids=[c['chunk_id'] for c in chunks],answer_claims=[dict(claim_id='c',text='事实')])
        client=FixtureClient('judge');original=client.ask
        def ask(task,instruction,data):
            if task=='answer_coverage':return dict(claims=[dict(claim_id='c',covered=True,supported=True,citations=['e99'])])
            return original(task,instruction,data)
        client.ask=ask
        with self.assertRaisesRegex(ValueError,'unknown catalogue'):
            verify_necessity(qa,{c['chunk_id']:c for c in chunks},client)

    def test_comparison_identical_values_do_not_establish_contrast(self):
        chunks=chunk_documents(fixture_documents())[:2]
        facts=[]
        for i,c in enumerate(chunks):
            facts.append(validate_fact(dict(subject=str(i),predicate='属性',object='相同值',quote=c['text'],role='attribute',conditions=[],time='',polarity='affirmed',modality='certain'),c))
        index=CandidateIndex(facts)
        paths=mine_paths(facts,[e for f in facts for e in index.recall(f)])
        self.assertFalse(any(p['task_type']=='comparison' for p in paths))

    def test_definition_context_preserves_contiguous_heading(self):
        from codes.multihop_qa.extraction import definition_context
        source='2.\n9.\n3\n铸态稀土金属 castingrareearthmetal\n为一种组织致密的稀土金属。'
        quote,hint=definition_context(source,'为一种组织致密的稀土金属。')
        self.assertEqual(hint,'铸态稀土金属')
        self.assertIn(quote,source)
        self.assertTrue(quote.startswith(hint))

    def test_definition_context_does_not_prepend_previous_sentence(self):
        from codes.multihop_qa.extraction import definition_context
        self.assertEqual(definition_context('之前是另一事实。\n企业甲采用技术。','企业甲采用技术。'),('企业甲采用技术。',''))

    def test_glossary_subject_must_match_original_heading(self):
        chunk=chunk_documents(fixture_documents())[0]
        chunk['text']='企业甲 company\n企业甲采用知识图谱。'
        client=StagedClient('知识图谱')
        with self.assertRaisesRegex(ValueError,'glossary term'):extract(chunk,client)

    def test_diagnostics_preserves_unknown_historical_fields_and_evidence(self):
        import tempfile,json
        from pathlib import Path
        from codes.multihop_qa.diagnostics import diagnose_runs
        with tempfile.TemporaryDirectory() as d:
            run=Path(d)/'run';run.mkdir()
            (run/'report.json').write_text(json.dumps(dict(stage='extraction',usage=[dict(task='old')])))
            chunk=chunk_documents(fixture_documents())[0]
            (run/'chunks.jsonl').write_text(json.dumps(chunk)+'\n')
            (run/'errors.jsonl').write_text(json.dumps(dict(chunk_id=chunk['chunk_id'],reason='failed'))+'\n')
            out=Path(d)/'out'; report=diagnose_runs([run],out)
            self.assertIsNone(report['runs'][0]['attempted'])
            self.assertEqual(report['runs'][0]['calls_with_full_response'],0)
            self.assertEqual(report['runs'][0]['calls_missing_timing'],1)
            failure=json.loads((out/'failures.jsonl').read_text(encoding='utf-8'))
            self.assertEqual(failure['evidence'][0]['text'],chunk['text'])
            self.assertEqual(report['human_labels_created'],0)

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
