"""Create numeric-first viewer for validated, held-out XAI outputs."""
from textwrap import dedent
import nbformat as nb
from xai_common import ROOT


def md(s): return nb.v4.new_markdown_cell(dedent(s).strip())
def code(s): return nb.v4.new_code_cell(dedent(s).strip())


def main():
    cells=[md('''
    # 09. Explainable AI — Kinematic và Fusion Kinematic+CWT

    **258 participant, group, 6 outer folds, seed 42.** Notebook chỉ đọc kết quả,
    không fit/chọn model, feature, task, threshold hay gate. Mọi attribution là outer Test.
    Primary TreeSHAP ở **raw log-odds**, không phải xác suất; signed sum và absolute mass khác nhau.
    Các case theo nhãn/outcome chỉ dùng minh họa sau đánh giá. XAI không phải bằng chứng nhân quả/lâm sàng.
    '''),code('''
    from pathlib import Path
    import json
    import numpy as np
    import pandas as pd
    from IPython.display import display,Image,Markdown
    ROOT=Path.cwd() if (Path.cwd()/'results/xai').exists() else Path.cwd().parent
    OUT=ROOT/'results/xai'
    validation=json.loads((OUT/'validation.json').read_text())
    assert validation.get('numerical_validation_passed'), 'Run report_xai.py first'
    features=pd.read_csv(OUT/'feature_importance.csv')
    groups=pd.read_csv(OUT/'group_importance.csv')
    tasks=pd.read_csv(OUT/'task_importance.csv')
    fusion=pd.read_csv(OUT/'fusion_participants.csv')
    permutation=pd.read_csv(OUT/'permutation_importance.csv')
    mapping=pd.read_csv(OUT/'feature_mapping.csv')
    manifest=json.loads((OUT/'figures/manifest.json').read_text())
    pd.set_option('display.max_columns',None)
    pd.set_option('display.float_format','{:.6f}'.format)
    print('Read-only XAI. Raw score verified as log-odds for this binary LightGBM.')
    '''),md('''
    ## Top Features — bảng số

    Mean |SHAP| trên 258 outer Test rows, đơn vị log-odds. CSV feature_importance.csv có đủ 1.036 cột.
    top_k_fold_count là số fold thuộc Top-20, không phải tiêu chí chọn lại feature.
    '''),code('''
    display(features[['rank','task','feature','group','sum_mean_abs_shap','fold_sd_mass',
                      'attribution_mass_share_percent','mean_rank','sd_rank','top_k_fold_count']].head(30))
    '''),md('''
    ## Top Feature Groups — tổng mass và net contribution

    12 nhóm từ prefix thật trong features.py. Mass = E[Σ|φ|]; net magnitude = E[|Σφ|].
    Nhóm có số cột khác nhau: xem cả mass/n_features. Đây là aggregation của individual SHAP,
    không phải Group Shapley tính bằng joint masker.
    '''),code('''
    display(groups)
    '''),md('''
    ## Top Tasks và grouped permutation — bảng số đầy đủ

    Task order được xác minh từ schema/prepare và tái dựng raw source. Cả 14 task có 74 features.
    Permutation drop = baseline metric − metric sau hoán vị, đơn vị điểm phần trăm; số âm được giữ.
    '''),code('''
    display(tasks.merge(permutation[permutation.level=='task'],left_on='name',right_on='block',validate='one_to_one'))
    display(pd.read_csv(OUT/'shap_permutation_concordance.csv'))
    '''),md('''
    ## Fold Stability

    Spearman giữa các thứ hạng importance và Jaccard Top-k. Train folds chồng lấp;
    SD fold không phải CI và 15 cặp fold không phải 15 mẫu độc lập.
    '''),code('''
    display(pd.read_csv(OUT/'fold_stability_summary.csv'))
    display(pd.read_csv(OUT/'task_rank_stability.csv'))
    with pd.option_context('display.max_rows',None): display(pd.read_csv(OUT/'fold_stability.csv'))
    '''),md('''
    ## Kiểm chứng raw score / log-odds / probability

    Không sigmoid từng SHAP value. Probability SHAP có reference và đơn vị riêng.
    SHAP interventional dùng lossless encoding của các partition cây để tránh rounding threshold
    float32 của backend; mọi split, model output và masked coalition được bảo toàn, model gốc không đổi.
    Xem REPORT_VI.md và unit test ranh giới ngưỡng. Background chỉ lấy từ outer Train.
    '''),code('''
    checks=pd.read_csv(OUT/'output_space_checks.csv')
    display(checks.head(12))
    print('Max raw additivity error:',float((checks.raw_score-checks.raw_path_reconstructed).abs().max()))
    print('Max probability additivity error:',float((checks.probability-checks.probability_reconstructed).abs().max()))
    display(pd.read_csv(OUT/'explainer_sensitivity.csv'))
    display(pd.DataFrame(json.loads((OUT/'mapping_audit.json').read_text())['sentinels']))
    '''),md('''
    ## Fusion — cấu hình và corrected/harmed

    Late trộn xác suất với trọng số simplex; không có gate. Adaptive là confidence-gated probability
    correction: Δp_CWT = 1[|pK−0.5|≤τ] × α × (pC−pK). Không phải learned attention network.
    Δp so với baseline không phải SHAP của modality. Gate/weights không được chọn lại bằng XAI.
    '''),code('''
    display(pd.read_csv(OUT/'fusion_fold_mechanisms.csv'))
    display(pd.read_csv(OUT/'fusion_outcome_summary.csv'))
    with pd.option_context('display.max_rows',None): display(pd.read_csv(OUT/'fusion_corrected_harmed_cases.csv'))
    '''),md('''
    ## Mỗi participant: pK, pC, pFusion, weight, gate và Δp

    Đủ 516 hàng. Gate_status của Late là not_applicable, tránh hiểu nhầm Late có confidence gate.
    '''),code('''
    columns=['scenario','fold','participant_id','label','kinematic_probability','cwt_probability','fusion_probability',
             'effective_weight_kinematic','effective_weight_cwt','gate_status','gate_threshold','budget','delta_cwt','outcome']
    with pd.option_context('display.max_rows',None): display(fusion[columns].sort_values(['scenario','fold','participant_id']).reset_index(drop=True))
    '''),md('''
    ## CWT component attribution — chỉ ở embedding/PCA

    Model CWT là LogisticRegression. Linear contribution cộng về log-odds CWT, không phải probability
    contribution của Fusion. PC1 của hai fold không phải cùng một tọa độ; không gộp hoặc gán cho
    tác vụ/tín hiệu/ý nghĩa sinh học. Bảng giữ fold và reducer rõ ràng.
    '''),code('''
    components=pd.read_csv(OUT/'cwt_component_importance.csv')
    with pd.option_context('display.max_rows',None): display(components[components.reducer.str.startswith('pca')].reset_index(drop=True))
    '''),md('''
    ## Hình SHAP phục vụ paper

    Summary và Beeswarm là hai cách trình bày cùng dữ liệu, không phải hai kiểm chứng độc lập.
    Hình dependence là quan hệ quan sát X–SHAP; không phải hiệu ứng can thiệp.
    '''),code('''
    def show_figures(names):
        for name in names:
            item=next(x for x in manifest if x['name']==name)
            display(Markdown('**'+name+'** — '+item['description']+' Units: '+item['units']))
            display(Image(filename=str(OUT/'figures'/(name+'.png')),width=1100))
    show_figures(['shap_summary','shap_beeswarm','shap_bar','shap_dependence'])
    '''),md('''
    ## Local explanations — cả đúng, sai, corrected và harmed

    Case là participant đầu theo ID trong từng outcome Adaptive, chọn hậu nghiệm chỉ để minh họa.
    Waterfall giải thích Kinematic log-odds, không phải feature-level SHAP của Fusion.
    '''),code('''
    display(pd.DataFrame(json.loads((OUT/'local_cases.json').read_text())))
    show_figures(['waterfall_corrected','waterfall_harmed','waterfall_unchanged_correct','waterfall_unchanged_wrong','task_waterfall'])
    '''),md('''
    ## Task/group importance và stability
    '''),code('''
    show_figures(['task_importance','feature_groups','task_group_heatmap','task_fold_ranks','grouped_permutation_tasks'])
    '''),md('''
    ## CWT giúp hoặc gây hại khi nào
    '''),code('''
    show_figures(['fusion_probability_correction','fusion_corrected_harmed','cwt_pca_components'])
    display(pd.read_csv(OUT/'fusion_confidence_bins.csv'))
    '''),md('''
    ## Kết luận và giới hạn

    Không kết luận số task tối thiểu từ SHAP. Muốn giảm task cần thí nghiệm task ablation riêng,
    chọn subset trong inner Train và xác nhận trên outer Test/cohort ngoài. Không tối ưu Accuracy
    hoặc thay model trong notebook này. Đọc REPORT_VI.md cho khuyến nghị hình/bảng viết paper.
    '''),code('''
    display(pd.DataFrame([{k:v for k,v in validation.items() if k!='numeric_artifact_sha256'}]))
    print((OUT/'REPORT_VI.md').read_text(encoding='utf-8').split('## Mapping')[0])
    ''')]
    notebook=nb.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3 (DiaGraMo)','language':'python','name':'python3'}})
    nb.validate(notebook); nb.write(notebook,ROOT/'notebooks/09_xai.ipynb')
    print('Created notebooks/09_xai.ipynb')


if __name__=='__main__': main()
