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
    def test_extracted_line_wrapped_formula_prefix_is_rejected_before_graph(self):
        chunk=chunk_documents(fixture_documents())[0]
        chunk['text']='草酸稀土的化学式为REx(C2O4)\ny。'
        raw=dict(subject='草酸稀土',predicate='化学式为',object='REx(C2O4)',quote=chunk['text'],role='attribute',
                 conditions=[],time='',polarity='affirmed',modality='certain')
        with self.assertRaisesRegex(ValueError,'formula is truncated'):
            validate_fact(raw,chunk)

    def test_complete_line_wrapped_formula_is_allowed_in_fact(self):
        chunk=chunk_documents(fixture_documents())[0]
        chunk['text']='草酸稀土的化学式为REx(C2O4)\ny。'
        raw=dict(subject='草酸稀土',predicate='化学式为',object='REx(C2O4)y',quote=chunk['text'],role='attribute',
                 conditions=[],time='',polarity='affirmed',modality='certain')
        self.assertEqual(validate_fact(raw,chunk)['object'],'REx(C2O4)y')

    def test_comparison_enumeration_is_rejected_before_model_support(self):
        from codes.multihop_qa.generation import question_structure_issues
        qa=dict(task_type='comparison',question='甲和乙的属性分别是什么？',
                answer_claims=[dict(fact_ids=['a']),dict(fact_ids=['b'])])
        self.assertEqual(question_structure_issues(qa),[
            'enumerative_comparison_question','comparison_question_lacks_difference_goal',
            'comparison_answer_lacks_joint_claim'])

    def test_comparison_difference_with_joint_claim_is_eligible_for_model_checks(self):
        from codes.multihop_qa.generation import question_structure_issues
        qa=dict(task_type='comparison',question='甲和乙在相同条件下的属性差异是什么？',
                answer_claims=[dict(fact_ids=['a','b'])])
        self.assertEqual(question_structure_issues(qa),[])

    def test_noncomparison_is_not_rejected_by_comparison_language_rule(self):
        from codes.multihop_qa.generation import question_structure_issues
        qa=dict(task_type='entity_bridge',question='甲和乙分别是什么？',answer_claims=[dict(fact_ids=['a'])])
        self.assertEqual(question_structure_issues(qa),[])

    def test_line_wrapped_formula_suffix_cannot_be_dropped(self):
        from codes.multihop_qa.verification import verify_support
        qa=dict(answer_claims=[dict(claim_id='c',text='化学式是REx(C2O4)',evidence_chunk_ids=['e'])],evidence_chunk_ids=['e'])
        client=FixtureClient('judge')
        with patch.object(client,'ask') as ask:
            check=verify_support(qa,{'e':dict(text='化学式REx(C2O4)\ny 表示。')},client)
        self.assertFalse(check['passed']);ask.assert_not_called()
        self.assertEqual(check['deterministic_issues'][0]['source_fragment'],'REx(C2O4)y')

    def test_complete_formula_and_non_formula_claims_are_unchanged(self):
        from codes.multihop_qa.verification import formula_truncations
        qa=dict(answer_claims=[dict(claim_id='c',text='化学式REx(C2O4)y',evidence_chunk_ids=['e'])])
        self.assertEqual(formula_truncations(qa,{'e':dict(text='化学式REx(C2O4)\ny 表示。')}),[])

    def test_path_schema_requires_complete_verdict_before_ending(self):
        from codes.multihop_qa.response_schema import response_schema
        schema=response_schema('path_verification',dict(facts=[dict(fact_id='f')]))
        self.assertIn('facts',schema['required'])
        self.assertIn('common_target',schema['required'])
        row=schema['properties']['facts']['items']
        self.assertEqual(row['properties']['fact_id']['enum'],['f'])
        self.assertIn('qualifiers_complete',row['required'])
        self.assertIsNone(response_schema('unknown',{}))

    def test_generation_schema_allows_abstention_without_forcing_an_answer(self):
        from codes.multihop_qa.response_schema import response_schema
        schema=response_schema('generation',dict(path=dict(fact_ids=['f'])))
        choices=schema['properties']['qa']['anyOf']
        self.assertIn(dict(type='null'),choices)
        answer=choices[1]
        self.assertIn('answer_claims',answer['required'])
        self.assertEqual(answer['properties']['answer_claims']['items']['properties']['fact_ids']['items']['enum'],['f'])

    def test_constraint_tokenizer_retains_unicode_and_skips_special_tokens(self):
        from codes.multihop_qa.constrained_decoding import regular_tokens
        class Tokenizer:
            all_special_ids=[2]
            def __len__(self):return 3
            def encode(self,value,add_special_tokens=False):return [0]
            def decode(self,ids):return ''.join({0:'0',1:'甲',2:'<eos>'}[i] for i in ids)
        self.assertEqual(list(regular_tokens(Tokenizer())),[(0,'0',False),(1,'甲',False)])

    def test_coverage_schema_limits_references_but_not_semantic_booleans(self):
        from codes.multihop_qa.response_schema import response_schema
        schema=response_schema('answer_coverage',dict(target_claims=[dict(claim_id='c1')],allowed_citations=[dict(citation_id='e0')]))
        row=schema['properties']['claims']['items']
        self.assertEqual(row['properties']['citations']['items']['enum'],['e0'])
        self.assertEqual(row['properties']['claim_id']['enum'],['c1'])
        self.assertEqual(row['properties']['supported'],dict(type='boolean'))
        self.assertEqual(schema['properties']['claims']['maxItems'],1)

    def test_coverage_schema_rejects_absent_evidence_catalogue(self):
        from codes.multihop_qa.response_schema import response_schema
        with self.assertRaises(ValueError):response_schema('answer_coverage',dict(target_claims=[dict(claim_id='c')],allowed_citations=[]))

    def test_local_decoding_mode_is_explicit_and_audited(self):
        from codes.multihop_qa.providers import LocalJSONClient
        with self.assertRaises(ValueError):LocalJSONClient('unused',json_mode='fallback')
        client=LocalJSONClient('unused',json_mode='unconstrained')
        with patch.object(client,'_ask',return_value={}):client.ask('test','prompt',{})
        self.assertEqual(client.history[0]['parameters']['json_mode'],'unconstrained')

    def test_schema_mode_does_not_salvage_trailing_junk_after_generation(self):
        from codes.multihop_qa.providers import parse_model_object
        with self.assertRaises(ValueError):parse_model_object('{"claims": []}"')

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
