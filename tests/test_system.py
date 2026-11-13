import copy
import tempfile
import unittest
from pathlib import Path
from codes.multihop_qa.ingestion import document_from_text, chunk_documents, parse_file
from codes.multihop_qa.extraction import validate_fact, validate_cached_fact
from codes.multihop_qa.retrieval import CandidateIndex, VectorLSH, compatible
from codes.multihop_qa.mining import mine_paths
from codes.multihop_qa.graph import build_graph
from codes.multihop_qa.verification import validate_qa, verify_support, verify_necessity, verify_path
from codes.multihop_qa.generation import generate_qa
from codes.multihop_qa.export import export_sft, split_records
from codes.multihop_qa.schema import empty_output, write_jsonl
from codes.multihop_qa.pipeline import extraction_stage, mining_stage, load
from codes.multihop_qa.demo import fixture_documents, fixture_facts, FixtureClient, run_demo
from codes.multihop_qa.training import train_scorer, predict


class RebuiltSystemTests(unittest.TestCase):
    def fixture(self):
        docs = fixture_documents()
        chunks = chunk_documents(docs)
        facts = [validate_fact(f, c) for f, c in zip(fixture_facts(chunks), chunks)]
        index = CandidateIndex(facts)
        edges = [e for f in facts for e in index.recall(f)]
        paths = mine_paths(facts, edges)
        path = next(p for p in paths if p['task_type'] == 'problem_solution')
        return docs, chunks, facts, edges, path

    def qa(self):
        _, chunks, facts, _, path = self.fixture()
        by_fact = {f['fact_id']: f for f in facts}
        by_chunk = {c['chunk_id']: c for c in chunks}
        qa, rejection = generate_qa(path, by_fact, by_chunk, FixtureClient('generator'), FixtureClient('judge'))
        self.assertIsNone(rejection)
        return qa, chunks, facts, path

    def test_heading_does_not_discard_body(self):
        docs, chunks, *_ = self.fixture()
        self.assertEqual(len(chunks), 3)
        self.assertTrue(all(not c['needs_review'] for c in chunks))
        self.assertEqual(chunks[0]['section'], ['企业技术'])
        self.assertNotIn('#', chunks[0]['text'])

    def test_heading_without_blank_line(self):
        doc = document_from_text('# 标题\n正文有完整内容。')
        chunks = chunk_documents([doc])
        self.assertEqual(chunks[0]['text'], '正文有完整内容。')
        self.assertEqual(chunks[0]['section'], ['标题'])

    def test_exact_provenance_after_splitting(self):
        doc = document_from_text('甲' * 13 + '。乙' * 20)
        chunks = chunk_documents([doc], 9)
        self.assertEqual(''.join(c['text'] for c in chunks), doc['blocks'][0]['text'])
        for c in chunks:
            span = c['spans'][0]
            self.assertEqual(doc['blocks'][0]['text'][span['start']:span['end']], c['text'])

    def test_stable_ids_and_duplicate_input(self):
        docs, chunks, *_ = self.fixture()
        self.assertEqual(chunk_documents(docs + docs), chunks)

    def test_exact_copy_source_groups(self):
        a = document_from_text('# A\n\n相同正文。')
        b = document_from_text('# B\n\n相同正文。')
        self.assertNotEqual(a['doc_id'], b['doc_id'])
        self.assertEqual(a['source_group_id'], b['source_group_id'])

    def test_fact_tampering_fails(self):
        _, chunks, facts, *_ = self.fixture()
        for changes in ({'quote': '编造'}, {'start': 99}, {'polarity': 'unknown'}, {'conditions': None}, {'role': 'fake'}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                validate_fact(dict(facts[0], **changes), chunks[0])

    def test_cached_canonical_id_tampering_fails(self):
        _, chunks, facts, *_ = self.fixture()
        f = dict(facts[0], subject_id='altered')
        with self.assertRaises(ValueError):
            validate_cached_fact(f, chunks[0])

    def test_explicit_alias_mapping(self):
        _, chunks, *_ = self.fixture()
        f = validate_fact(fixture_facts(chunks)[0], chunks[0], {'企业甲': 'company:001'})
        self.assertEqual(f['subject_id'], 'company:001')

    def test_role_recall_and_three_evidence_structure(self):
        _, _, facts, edges, path = self.fixture()
        self.assertEqual(len(path['fact_ids']), 3)
        self.assertTrue(path['cross_document'])
        self.assertTrue(any('complement' in e['channels'] for e in edges))
        self.assertIn('complementarity', path['score_components'])

    def test_no_unrelated_path(self):
        _, _, facts, *_ = self.fixture()
        for i, f in enumerate(facts):
            f['subject_id'] = f'unrelated{i}'
            f['object_id'] = f'distinct{i}'
        index = CandidateIndex(facts)
        self.assertFalse(mine_paths(facts, [e for f in facts for e in index.recall(f)]))

    def test_conflicting_conditions_pruned(self):
        _, _, facts, *_ = self.fixture()
        facts[0]['conditions'] = ['mode=A']
        facts[1]['conditions'] = ['mode=B']
        self.assertFalse(compatible(facts[0], facts[1]))

    def test_negation_and_uncertainty_not_positive_path(self):
        for key, value in [('polarity', 'negated'), ('modality', 'uncertain')]:
            _, _, facts, *_ = self.fixture()
            facts[0][key] = value
            index = CandidateIndex(facts)
            paths = mine_paths(facts, [e for f in facts for e in index.recall(f)])
            self.assertFalse(any(p['task_type'] == 'problem_solution' for p in paths))

    def test_global_order_is_input_order_independent(self):
        _, _, facts, edges, _ = self.fixture()
        self.assertEqual(mine_paths(facts, edges, path_limit=1), mine_paths(list(reversed(facts)), edges, path_limit=1))

    def test_bfs_control_and_invalid_budget(self):
        _, _, facts, edges, _ = self.fixture()
        self.assertTrue(mine_paths(facts, edges, strategy='bfs'))
        with self.assertRaises(ValueError):
            mine_paths(facts, edges, beam_width=0)

    def test_four_step_bridge(self):
        facts = []
        for i in range(4):
            c = chunk_documents([document_from_text(f'实体{i}连接实体{i+1}。')])[0]
            f = dict(chunk_id=c['chunk_id'], subject=f'实体{i}', predicate='连接', object=f'实体{i+1}',
                     role='method', quote=c['text'], conditions=[], time='', event='', polarity='affirmed', modality='certain')
            facts.append(validate_fact(f, c))
        index = CandidateIndex(facts)
        paths = mine_paths(facts, [e for f in facts for e in index.recall(f)])
        self.assertTrue(any(p['task_type'] == 'entity_bridge_4' and p['candidate_steps'] == 4 for p in paths))

    def test_optional_vector_recall(self):
        index = VectorLSH({'a': [1., 0.], 'b': [1., 0.], 'c': [0., 1.]})
        self.assertEqual(index.query('a', 1)[0], ('b', 1.))
        for vectors in ({'a': [0., 0.]}, {'a': [1.], 'b': [1., 2.]}, {'a': [float('nan')]}):
            with self.assertRaises(ValueError):
                VectorLSH(vectors)

    def test_multigranularity_graph(self):
        docs, chunks, facts, edges, _ = self.fixture()
        graph = build_graph(docs, chunks, facts, edges)
        self.assertTrue({'Document', 'Chunk', 'Sentence', 'Fact', 'Entity'} <= {n['type'] for n in graph['nodes']})
        self.assertTrue(any(e['type'] == 'RELATION_COMPLEMENT' and e['status'] == 'candidate' for e in graph['edges']))

    def test_empty_claim_rejected(self):
        _, _, facts, _, path = self.fixture()
        with self.assertRaises(ValueError):
            validate_qa(dict(question='问题', answer='答案', answer_claims=[dict(text='', fact_ids=path['fact_ids'])]), path, {f['fact_id']: f for f in facts})

    def test_uncovered_answer_rejected(self):
        qa, _, facts, path = self.qa()
        raw = dict(question=qa['question'], answer=qa['answer'] + '因此技术是最优选择。',
                   answer_claims=[dict(text=c['text'], fact_ids=c['fact_ids']) for c in qa['answer_claims']])
        with self.assertRaises(ValueError):
            validate_qa(raw, path, {f['fact_id']: f for f in facts})

    def test_missing_claim_review_rejected(self):
        qa, chunks, *_ = self.qa()
        class Missing(FixtureClient):
            def ask(self, task, instruction, data):
                return dict(claims=[])
        with self.assertRaises(ValueError):
            verify_support(qa, {c['chunk_id']: c for c in chunks}, Missing('judge'))

    def test_path_failure_stops_generation(self):
        _, chunks, facts, _, path = self.fixture()
        class Reject(FixtureClient):
            def ask(self, task, instruction, data):
                r = super().ask(task, instruction, data)
                r['relations_valid'] = False
                return r
        generator = FixtureClient('generator')
        qa, rejection = generate_qa(path, {f['fact_id']: f for f in facts}, {c['chunk_id']: c for c in chunks}, generator, Reject('judge'))
        self.assertIsNone(qa)
        self.assertEqual(generator.history, [])

    def test_all_group_subsets_evaluated(self):
        qa, chunks, *_ = self.qa()
        result = verify_necessity(qa, {c['chunk_id']: c for c in chunks}, FixtureClient('judge'))
        self.assertTrue(result['passed'])
        self.assertEqual(len(result['trials']), 8)
        self.assertTrue(any(not r['evidence_chunk_ids'] for r in result['trials']))

    def test_single_evidence_shortcut_rejected(self):
        qa, chunks, *_ = self.qa()
        class Shortcut(FixtureClient):
            def ask(self, task, instruction, data):
                if task == 'answer_coverage' and data['evidence']:
                    c = data['evidence'][0]
                    return dict(claims=[dict(claim_id=t['claim_id'], covered=True, supported=True,
                                            citations=[dict(chunk_id=c['chunk_id'], quote=c['text'])]) for t in data['target_claims']])
                return super().ask(task, instruction, data)
        self.assertFalse(verify_necessity(qa, {c['chunk_id']: c for c in chunks}, Shortcut('judge'))['passed'])

    def test_absent_evidence_citation_rejected(self):
        qa, chunks, *_ = self.qa()
        class Fabricate(FixtureClient):
            def ask(self, task, instruction, data):
                if task == 'answer_coverage':
                    return dict(claims=[dict(claim_id=t['claim_id'], covered=True, supported=True,
                                            citations=[dict(chunk_id='absent', quote='编造')]) for t in data['target_claims']])
                return super().ask(task, instruction, data)
        with self.assertRaises(ValueError):
            verify_necessity(qa, {c['chunk_id']: c for c in chunks}, Fabricate('judge'))

    def test_connected_sources_never_leak(self):
        rows = [dict(qa_id='a', source_group_ids=['1', '2']), dict(qa_id='b', source_group_ids=['2', '3']), dict(qa_id='c', source_group_ids=['3'])]
        self.assertEqual(len(set(split_records(rows).values())), 1)

    def test_new_qa_exports_without_legacy_plan(self):
        qa, chunks, *_ = self.qa()
        review = dict(qa_id=qa['qa_id'], decision='approved', human_reviewed=True, reviewer='test-reviewer')
        with tempfile.TemporaryDirectory() as root:
            report = export_sft([qa], {c['chunk_id']: c for c in chunks}, [review], Path(root) / 'export')
            self.assertEqual(report['exported'], 1)

    def test_post_verification_answer_change_not_exported(self):
        qa, chunks, *_ = self.qa()
        qa['answer'] += '未经核验的内容。'
        review = dict(qa_id=qa['qa_id'], decision='approved', human_reviewed=True, reviewer='test-reviewer')
        with tempfile.TemporaryDirectory() as root:
            report = export_sft([qa], {c['chunk_id']: c for c in chunks}, [review], Path(root)/'export')
            self.assertEqual(report['exported'], 0)

    def labelled_pairs(self):
        return [dict(fact_ids=[f'{split}-{label}-a', f'{split}-{label}-b'],
                     features=dict(entity=.3, relation=float(label), semantic=.3, complement=float(label)),
                     label=label, source_group_ids=[split], split=split, human_reviewed=True, reviewer='test')
                for split in ('train', 'val') for label in (0, 1)]

    def test_trainable_complement_scorer(self):
        import json
        with tempfile.TemporaryDirectory() as root:
            report = train_scorer(self.labelled_pairs(), Path(root)/'scorer', epochs=100)
            model = json.loads((Path(root)/'scorer'/'scorer.json').read_text(encoding='utf8'))
            rows = self.labelled_pairs()
            self.assertGreater(predict(rows[1]['features'], model), predict(rows[0]['features'], model))
            self.assertEqual(report['metrics']['val']['samples'], 2)

    def test_training_source_leakage_rejected(self):
        rows = self.labelled_pairs()
        rows[-1]['source_group_ids'] = ['train']
        with tempfile.TemporaryDirectory() as root, self.assertRaises(ValueError):
            train_scorer(rows, Path(root)/'scorer')

    def test_docx_heading_and_table(self):
        import zipfile
        with tempfile.TemporaryDirectory() as root:
            path = Path(root)/'example.docx'
            xml = '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body><w:p><w:pPr><w:pStyle w:val="Heading1"/></w:pPr><w:r><w:t>标题</w:t></w:r></w:p><w:p><w:r><w:t>正文信息。</w:t></w:r></w:p><w:tbl><w:tr><w:tc><w:p><w:r><w:t>单元格</w:t></w:r></w:p></w:tc></w:tr></w:tbl></w:body></w:document>'
            with zipfile.ZipFile(path, 'w') as archive:
                archive.writestr('word/document.xml', xml)
            chunks = chunk_documents([parse_file(path)])
            self.assertEqual(chunks[0]['section'], ['标题'])
            self.assertFalse(chunks[0]['needs_review'])
            self.assertTrue(chunks[1]['needs_review'])

    def test_html_scripts_excluded(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root)/'example.html'
            path.write_text('<script>不应出现</script><p>正文内容。</p>', encoding='utf8')
            doc = parse_file(path)
            self.assertNotIn('不应出现', ''.join(b['text'] for b in doc['blocks']))

    def test_pdf_text_layer_marked_for_review(self):
        try:
            import pymupdf
        except ImportError:
            self.skipTest('optional PDF dependency not installed')
        with tempfile.TemporaryDirectory() as root:
            path = Path(root)/'example.pdf'
            with pymupdf.open() as pdf:
                page = pdf.new_page()
                page.insert_text((72,72), 'Evidence with a native text layer.')
                pdf.save(path)
            chunks = chunk_documents([parse_file(path)])
            self.assertEqual(chunks[0]['source_position']['pages'], [1])
            self.assertTrue(chunks[0]['needs_review'])

    def test_human_release_gate_and_demo_gate(self):
        qa, chunks, *_ = self.qa()
        with tempfile.TemporaryDirectory() as root:
            report = export_sft([qa], {c['chunk_id']: c for c in chunks}, [], Path(root) / 'export')
            self.assertEqual(report['exported'], 0)
            qa['demo_only'] = True
            review = dict(qa_id=qa['qa_id'], decision='approved', human_reviewed=True, reviewer='test')
            report = export_sft([qa], {c['chunk_id']: c for c in chunks}, [review], Path(root) / 'demo')
            self.assertEqual(report['exported'], 0)

    def test_existing_output_not_overwritten(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root)
            (path / 'user.txt').write_text('保留', encoding='utf8')
            with self.assertRaises(ValueError):
                empty_output(path)

    def test_offline_end_to_end(self):
        with tempfile.TemporaryDirectory() as root:
            result = run_demo(Path(root) / 'demo')
            self.assertGreater(result['candidate_paths'], 0)
            self.assertGreater(result['fixture_auto_passed'], 0)
            self.assertEqual(result['errors'], 0)
            self.assertEqual(result['production_exported'], 0)


if __name__ == '__main__':
    unittest.main()
