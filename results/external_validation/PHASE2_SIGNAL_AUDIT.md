# Phase 2 — Audit raw và đơn vị độc lập

Đã khảo sát toàn bộ 121 SVC. 120 recording qua kiểm tra header/7columns/finite values; 1 file rỗng bị reader từ chối. Vì vậy all-files-pass=False, không giả đánh dấu toàn bộ là hợp lệ. Một người có file rỗng và một recording hợp lệ khác, nên vẫn có120người với tín hiệu. Không đổi source hay tự xóa bản ghi. File không có fieldnames. Phân tích theo c0…c6 và gọi timestamp/pen-state **candidate**, không tự gán physical units.

Candidate c2 có median positiveΔt=8.0 native units, zeroΔt=0, negativeΔt=2; candidate c3 binary ở mọi recording=True. Khoảng nhảy>10×median riêngrecording chỉ là diagnostic, không chia hay resample dữ liệu. Áp suất/góc chỉ audit giá trị số/vùng theo cột; không chuyển altitude→tilt hoặc ticks→seconds.

Excel–SVC join đủ120ID, labelsbinary, một IDcó2recordings. Đã tạo candidate5folds seed42, mọi recordingscùngngười cùngfold; không có sample/window randomsplit. Đây là grouping theo IDnguồn, không bằng chứng độc lập vật lý tuyệt đối. XML history/comments có thể nhạy cảm; chỉ kiểm tra trùng nonempty trong phần join được, không publicnội dung.

Các bảng recording/participant và candidatefolds chỉ ở `data/external_datasets/drotar_dobes_2020/`. Publicchannelstatistics trung bình recordingtrongngười trước khi tổng hợp; không coi hàng triệu samples là replicates. Không có city/site/device serial/date theo participant, không tạo source bằng cách sao label sang source. Sourceindependence vẫn UNVERIFIED. Metadataage/sex/hand không dùng làm predictors.

**PARTIAL:** structural/grouping audit hoàn tất, nhưng exactschema/units và taskboundaries chưa xác minh, nên chưa đủ cho physical harmonization hay clinical/raw-model comparison.
