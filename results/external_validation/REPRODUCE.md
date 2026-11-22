# Reproduce — bounded Phase6 investigation

Chạy từ projectroot bằng virtualenv hiện tại; không chạy prepare.py, evaluate.py, báo cáo cũ hoặc fullPhase4queue.

```powershell
.venv/Scripts/python.exe -X utf8 scripts/external_validation/run.py --phase all --max-runtime-hours 8 --report-reserve-minutes 15
```

Các phase CLI:0,1,continue,validate. All/continue kiểm tra lockedrecipe và resumes priorfolds đã hoàn tất, không refit. SESSION_CLOCK giữ mốc toàn session qua các CLI invocation; không reset budget bằng việc tách lệnh. Muốn một session mới sau khi hết giờ cần protocol vận hành riêng, không tự xóa clock để kéo dài budget. Budgetreserve phục vụ tests/reporting, không fillGPUtime.

Nguồnpincommit0079fdd34fc5eff0c1c58f9689127b2c73b57f8b, files bounded≤32MiB, Gitblob/SHA256/sizechecks, ZIPpath/CRC/size/symlink/executableguard. Filesđãcó phải khớp hash; extractexistingfiles khác bị từ chối. Binaryfetch dùng publicrawGitHub vì connector chỉUTF8. Download permissions chỉ cho localread theo task; không redistribution.

Excel được đọc OOXML bằng stdlib: giữrawcachedvalues, không executeformulas. Một directlabelreference được đối chiếu cachedvalue. IDnumeric và text được canonicalizezero-padding label-independent; collision/missinglabels bị từ chối. Không dùng XMLcommenthistory làm feature/label; chỉ locally kiểm tra nguồn.

ReaderSVC độc lập, trả nativec0…c6, rejects0samples/nonfinite/header mismatch; không gọi baselinepreprocessing. No time/angleconversion/windowing/scaler/rawtraining. Candidateclock/pen diagnostics không phải verifiedphysicalschema.

LOCKED_CONFIG được tạo trướcscores; priorDummyfit Train-only,5participantfolds,seed42,thresholdp>0.5. JSONclasscountcheckpoint và SHA/split/configsignatures; no optimizer/RNG N/Aclosed-form. IndividualOOF/splits/metadata/checkpoints trong data/external_datasets đãGitignore. Publictables chỉaggregatefold/cohortresults. Không fusion hoặc Kinematic/Temporalfit.

PaperHTML/Supplement được kiểm tra bằng publicprimaryURLs trong SOURCE_DOCUMENTS.csv; caches chỉlocal. Supplement1trang render và xem trực tiếp; không suy taskboundaries từ ảnh. Reproduce từclonekhác cần tài liệu cached hoặc checkURLlại; no automatic access to request-onlyFrenchcohorts.

Tests/validation/manifests tạo dưới results/external_validation. Finalintegrityhash kiểm tra toàn protectedmanifest, checkpointdata + sourcefiles. Không sửa baseline/notebooks/oldreports, không commit/push/merge. Các mụcNOT_RUN không được đổi thànhPASS.
