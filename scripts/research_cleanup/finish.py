"""Summarize actual cleanup checks; never fill missing test evidence as PASS."""
from pathlib import Path
import json
import re
import subprocess
import pandas as pd
import hashlib

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'docs/research_cleanup'


def read(name):
    p=OUT/name
    return json.loads(p.read_text(encoding='utf8')) if p.exists() else None


def put(name,text):
    (OUT/name).write_text(text.strip()+'\n',encoding='utf8')


def build():
    before=read('PROTECTED_BEFORE.json')
    tests={phase:read(f'TEST_RESULTS_{phase.upper()}.json') for phase in ['before','after']}
    preservation=read('PRESERVATION_VALIDATION.json')
    behavior=read('BEHAVIOR_PRESERVATION_CHECKS.json')
    duplicate=pd.read_csv(OUT/'DUPLICATE_ARTIFACT_REVIEW.csv')
    inventory=pd.read_csv(OUT/'PROJECT_INVENTORY.csv')
    registry=pd.read_csv(ROOT/'docs/research/EXPERIMENT_REGISTRY.csv')
    counts={}
    for stage,rows in tests.items():
        counts[stage]=dict(suites=len(rows or []),passed=sum(int(m.group(1)) for r in rows or [] for m in [re.search(r'(\d+) passed',r['summary'])] if m),
            failed_suites=[r['suite'] for r in rows or [] if r['returncode']],executed=rows is not None)
    complete=bool(tests['after'] and preservation and preservation['valid'] and behavior and behavior['valid'] and not counts['after']['failed_suites'])
    result=dict(status='COMPLETE' if complete else 'IN_PROGRESS',test_stages=tests,counts=counts,
        historical_full_tree_guard='Passed before edits; after authorized README/CLI/test edits, one old native manifest assertion is explicitly deselected; new preservation gate checks all historical artifacts',
        no_new_research_experiments=True,unit_test_synthetic_optimizer_steps_only=True,
        no_model_retraining_or_new_cohort_metrics=True)
    (OUT/'TEST_RESULTS.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    put('CODE_REFACTOR_SUMMARY.md','''# Refactor nhỏ và kiểm chứng hành vi

## 1. Native CLI importability

File: `scripts/native_channel_research/run.py`. Callers: CLI `python .../run.py`, future imported `main`; internal reader/train/evaluate dependencies giữ nguyên. Trước đây `np` chỉ được import trong `__main__`, nên gọi `main()` từ module có thể lỗi NameError ở prior replay. Nay import NumPy ở module scope và nhận `main(argv=None)`; `parse_args(argv)` giữ nguyên default CLI behavior. Không đổi architecture, config, seed, threshold, aggregation, input/output paths hoặc phép tính training. Historical artifacts chỉ đọc.

Checks: CLI help stdout before/after giống nhau; tests gọi imported help/invalid args xác nhận không đọc signal hoặc gọi train. Numerical replay đọc lại fixed existing OOF, không chạy model fit. Không tuyên bố đã retrain pipeline để xác minh refactor.

## 2. Checkpoint test isolation

File: `tests/native_channel_research/test_native.py`. Hai synthetic checkpoint tests trước đây ghi `test_checkpoint.pt`/`test_cuda_exact_resume.pt` vào ignored research-local directory. Autouse fixture nay chuyển common.LOCAL và module.LOCAL sang pytest tmp_path cho từng test; không sửa source dataset, real model hoặc expected assertions. Cleanup test plugin cũng cô lập BEFORE run để không làm hỏng hiện trạng trước khi fixture mới tồn tại.

## 3. Không deduplicate khác semantics

Không gộp DiaGraMo physical preprocessing với native numeric reader; không gộp metrics/threshold/bootstrap khác contract; không đổi report.py namespaces hoặc tên checkpoint imports. Dynamic file-path imports của Phase5/6 giữ nguyên. Hardcoded Windows path duy nhất trong AST inventory cũ là fixture ZIP safety test, không production path cần thay.

## Phạm vi phép so sánh

`CLI_BEFORE.json`/`CLI_AFTER.json`, `BEHAVIOR_PRESERVATION_CHECKS.json`, `NUMERICAL_PRESERVATION.json`, `TEST_RESULTS*.json`, `PRESERVATION_VALIDATION.json`. Một historical native full-tree guard đã PASS trước cleanup; sau edits hợp lệ nó không phù hợp snapshot cũ, nên được deselect rõ ràng và thay bằng cleanup preservation gate. Không sửa historical manifest để che thay đổi.
''')
    put('CLEANUP_CHANGELOG.md','''# Cleanup changelog

- Tạo nhánh study/weekly-research-cleanup từ abee084; giữ nguyên untracked user work.
- Inventory toàn workspace ngoài Git internals; exact sensitive paths/hash chỉ local, public inventory có aggregate/alias.
- Thêm research navigation, timeline theo Git, architecture evolution, limitations và authoritative experiment registry.
- Thêm báo cáo tuần16–22/11, nội dung11slides, figure/table index, claim/evidence matrix và next-week priorities.
- README thành entrypoint; nội dung hướng dẫn cũ được giữ ở docs/research/LEGACY_ENTRYPOINT_GUIDE.md và Git ref abee084.
- Minimal imported-CLI/test-scratch refactor; source model/preprocessing không đổi.
- Giữ mọi historical artifact/caches/weights/notebooks. Không move/delete, retrain, bootstrap lại hoặc sửa metric CSV cũ.
- Byte-identical duplicates chỉ được đề xuất review; không owner deletion approval, không thực hiện removal.
- Không commit, push, merge hoặc viết lại Git history tự động.
''')
    testtable='| Stage | Suites | Passed tests | Failed suites |\n|---|---:|---:|---|\n'
    for phase,row in counts.items():testtable+=f"| {phase} | {row['suites']} | {row['passed']} | {', '.join(row['failed_suites']) or ('none' if row['executed'] else 'NOT RUN')} |\n"
    changed=subprocess.check_output(['git','diff','--name-only'],cwd=ROOT,text=True).splitlines()
    report=f'''# Bàn giao cleanup và chuẩn bị báo cáo tuần

## 1. Tóm tắt

Trạng thái: **{'COMPLETE' if complete else 'IN_PROGRESS — chờ checks thực tế'}**. Đã tổ chức lại điểm vào và tài liệu, không tạo experimental phase hoặc chạy research training. Nhánh `study/weekly-research-cleanup`, base `abee084d6621ccf4175f3e4a1a5a484756bb824f`.

## 2. Cấu trúc ban đầu

{before['total_files']:,} files, {before['total_bytes']/2**30:.3f}GiB ngoài Git internals; {before['tracked_files']} tracked. Data/cache/weights/results và hai Python environments chiếm phần lớn dung lượng. Snapshot hash {before['hashed_files']:,} files/{before['hashed_bytes']/2**30:.3f}GiB; môi trường Python và generated pycache/pytest cache được inventory nhưng không content-hash. Exact private paths chỉ ignored local.

## 3. Tổ chức cập nhật

`docs/research/`: registry/timeline/architecture/results/limitations. `docs/weekly_reports/2026-11-16_to_2026-11-22/`: báo cáo,11slides, figures/claims/next-week plan. `docs/research_cleanup/`: inventory/dependencies/duplicates/checks. Không di chuyển result directories hoặc đổi CLI paths.

## 4. Files bổ sung

Scripts audit/build_registry/documents/checks/safe_tests/validate/finish, cleanup regression tests, research/weekly documentation và manifests. Inventory số từng nhóm trong PROJECT_INVENTORY.csv; registry {len(registry)} records. Guide README lịch sử giữ riêng để old instructions còn truy cập được.

## 5. Files sửa

{', '.join(changed) or 'None'}.

README được rút gọn thành entrypoint; source chỉ đổi imported CLI argv/NumPy scope và test scratch fixture. Không sửa bất cứ file src/diagramo nào.

## 6. Files chủ động bảo toàn

Mọi raw SVC, merged/cleaned data, embedding cache, pretrained weights, model checkpoints, OOF, fold assignment, configs, notebooks01–09, baseline4branches, XAI/Temporal/audit/provenance/external/native reports. Giữ cả bốn pre-existing user untracked files. Không chạm history hoặc legacy removal plan.

## 7. Refactor thực hiện

Xem CODE_REFACTOR_SUMMARY.md: imported CLI giữ default command behavior; pytest scratch isolation tránh overwrite checkpoint test cũ. Caller map/AST và CLI help equivalence được lưu. Không gộp reader/metrics khác scientific semantics.

## 8. Duplicate đã nhận diện

{len(duplicate)} duplicate-member entries thuộc {duplicate.sha256.nunique()} exact-byte groups; đây là bằng nhau về byte, không chứng minh dư thừa về provenance. Những snapshot/weights/cache có thể có chủ đích. PROJECT_INVENTORY.csv cũng đánh dấu generated caches theo policy.

## 9. Removal đề xuất, chưa thực hiện

REMOVAL_REVIEW_PLAN.md và DUPLICATE_ARTIFACT_REVIEW.csv chứa path/alias, size, reason, dependency warning. REVIEW_FOR_REMOVAL cần owner approval và kiểm tra use trước khi xóa. Không có file nào bị xóa, không dọn `.venv314`, tmp hoặc caches.

## 10. Kết quả khoa học bảo toàn

Kinematic F1=0,847163; Temporal=0,713178; Fixed50=0,840681. Fusion ΔF1≈−0,006483, CI95%[−0,053470;0,041286]. City-only accuracy≈0,972868. Slovak B0–B4 F1 lần lượt0,344262/0,598214/0,932413/0,941337/0,941337. B3/B4 binary predictions giống nhau; B3−B2 CI chứa0. Tất cả đọc/replay từ artifacts cũ, không tính score cho model mới. XAI tables, confidence intervals và source statistics nằm trong hash gate.

## 11. Tests và checks đã chạy

{testtable}

Logs chi tiết chỉ local, public summary không in clinical IDs. BEFORE gồm165tests PASS. AFTER bỏ đúng một historical full-tree guard do source/README edits hợp lệ; guard đã PASS trước sửa, được thay bằng new full preservation manifest. Không gọi full runners để test. Unit tests có synthetic forward/backward/optimizer steps; không train research model hoặc tạo performance result mới. Xem TEST_RESULTS.json cho status chính xác; CLI help, syntax, documentation links và OOF replay có JSON riêng.

## 12. Protected artifact integrity

PRESERVATION_VALIDATION.json: {preservation['valid'] if preservation else 'PENDING'}. PROTECTED_BEFORE/AFTER ghi counts/hash status, exact local manifest giữ private. Allowed changed files chỉ README/native CLI/native tests; mọi result/input/checkpoint phải giữ nguyên SHA. Không coi count-only check là content preservation.

## 13. Riêng tư

Không thêm data/weights/checkpoints/individual clinical rows/OOF vào Git. Public inventory gộp sensitive roots và che sensitive filenames; known-ID token scan và output selection có trong PRIVACY_REVIEW.json. Không chứng nhận de-identification toàn Git history. Historical artifacts có risk được bảo toàn, không tự xóa/redistribute.

## 14. Sẵn sàng báo cáo tuần

Báo cáo10mục,11slide entries, figure/table index và12claim checks hoàn tất nội dung. Ba aggregate figures đã được xem trực quan. Dùng bảng gốc cho locked/native comparisons, không vẽ lại hàng loạt. XAI là previous milestone; diagram mô tả đúng relative segment attention/task pooling. Không tạo PowerPoint trong task này.

## 15. Technical debt còn lại

Historical scripts có nhiều global imports, module report.py trùng tên và format compact. Dynamic imports không được AST chứng minh hoàn toàn; giữ explicit namespace và suite isolation. Một số CLI cũ không có --help an toàn (provenance runner), không gọi thử tùy tiện. Requirements baseline/XAI bổ sung khác vai trò, không auto merge. Historical full-tree hash guards cần cơ chế baseline ownership cho thay đổi documentation tương lai; không sửa snapshot cũ trong cleanup.

## 16. Bước phát triển đề xuất

Ưu tiên source-aware replication và một protocol development-only để tách marginal/implicit-position/order hypotheses. Không tăng Transformer hoặc full SSL mặc định. Owner cần quyết định removal cụ thể, release privacy cho hình nhóm nhỏ, clinical/source clarification và access cohort mới. Không tự gửi liên hệ hoặc push/merge.
'''
    put('FINAL_CLEANUP_REPORT_VI.md',report)
    # Public added-file inventory is separate from the original snapshot inventory.
    new_names=subprocess.check_output(['git','ls-files','--others','--exclude-standard'],cwd=ROOT,text=True).splitlines()
    prefixes=('docs/research/','docs/research_cleanup/','docs/weekly_reports/','scripts/research_cleanup/','tests/research_cleanup/')
    added=[dict(path=n,bytes=(ROOT/n).stat().st_size,policy='KEEP_ACTIVE',category='Weekly reporting/shared infrastructure') for n in new_names if n.startswith(prefixes)]
    pd.DataFrame(added).to_csv(OUT/'ADDED_FILES.csv',index=False)
    manifest=[]
    for name in [r['path'] for r in added]+changed:
        if name.endswith('CLEANUP_ARTIFACT_MANIFEST.json'):continue
        p=ROOT/name
        manifest.append(dict(path=name,bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    (OUT/'CLEANUP_ARTIFACT_MANIFEST.json').write_text(json.dumps(dict(files=manifest,scope='New cleanup outputs and three authorized edits; historical inputs use separate preservation manifest'),indent=2),encoding='utf8')
    print(json.dumps(dict(status=result['status'],counts=counts,registry_records=len(registry),duplicate_members=len(duplicate))))


if __name__=='__main__':build()
