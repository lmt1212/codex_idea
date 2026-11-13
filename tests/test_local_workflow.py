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
