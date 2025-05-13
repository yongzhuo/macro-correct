# 过度训练实验
### acc_overfit(thr=0)
| model/acc                                                           | avg| sighan2013.dev.json| sighan2014.dev.json| sighan2015.dev.json| shibing624_CSC_test.json| lomo_tet.json| faspell.dev.json| lemon_v2.tet.json| ecspell.dev.json| mcsc_tet.json| csc_130w_to_de3.tet.json| pt_mask_4_passage_130w_10epoch.tet.json| text_proof.train.json| pt_passage_mask_and_mask_no_and_xxqg_3kw.tet.json| acc_rmrb.tet.json| acc_xxqg.tet.json |
|:--------------------------------------------------------------------|:-----------------|:-----------------|:-----------------|:-----------------|:-----------------|:-----------------|:-----------------|:-----------------|:-----------------|:-----------------|:-----------------|:-----------------|:-----------------|:-----------------|:-----------------|:-----------------|
| macbert4csc_pycorrector(pycorrector项目的官方版本macbert4csc)          | 87.58| 73.4| 84.93| 85.82| 85.82| 85.28| 90.8| 85.47| 92.2| 94.14| 82.63| 85.58| 89.7| 86.68| 94.39| 96.84 |
| wang271k_clean_std_of_macbert4mdcspell(全量MASK)                      | 85.49| 65.1| 81.54| 87.73| 87.73| 81.4| 90.5| 82.15| 89.4| 92.86| 80.81| 85.19| 86.73| 86.34| 89.6| 95.24 |
| wang271k_clean_std_of_macbert4mdcspell(随机MASK0.7文本, 0.30概率'正确->正确') | 91.74| 76.2| 88.7| 92.36| 92.27| 91.52| 94.2| 90.69| 95.0| 96.06| 87.39| 90.77| 94.19| 91.64| 96.64| 98.46 |
| wang271k_clean_std_of_macbert4mdcspell(随机MASK0.7文本, 0.15概率'正确->正确') | 90.3| 73.4| 87.1| 90.82| 90.73| 90.06| 93.0| 88.51| 93.93| 95.17| 85.79| 89.75| 91.98| 90.72| 95.0| 98.52 |


### acc_overfit(thr=0)
| model/acc_overfit                                                      | avg| sighan2013.dev.json| sighan2014.dev.json| sighan2015.dev.json| shibing624_CSC_test.json| lomo_tet.json| faspell.dev.json| lemon_v2.tet.json| ecspell.dev.json| mcsc_tet.json| csc_130w_to_de3.tet.json| pt_mask_4_passage_130w_10epoch.tet.json| text_proof.train.json| pt_passage_mask_and_mask_no_and_xxqg_3kw.tet.json| acc_rmrb.tet.json| acc_xxqg.tet.json |
|:-----------------------------------------------------------------------|:-----------------|:-----------------|:-----------------|:-----------------|:-----------------|:-----------------|:-----------------|:-----------------|:-----------------|:-----------------|:-----------------|:-----------------|:-----------------|:-----------------|:-----------------|:-----------------|
| acc_macbert4csc_pycorrector                                            | 87.58| 73.4| 84.93| 85.82| 85.82| 85.28| 90.8| 85.47| 92.2| 94.14| 82.63| 85.58| 89.7| 86.68| 94.39| 96.84 |
| acc_macbert4mdcspell_acc_by_add_true_thr7_thr85_36step                 | 93.88| 86.3| 89.92| 91.45| 91.0| 89.02| 95.3| 91.17| 98.4| 98.54| 93.83| 95.79| 95.37| 96.44| 96.25| 99.46 |
| acc_macbert4mdcspell_acc_by_add_true_thr7_thr85_72step                 | 94.17| 96.0| 92.31| 96.72| 99.58| 96.65| 96.54| 97.52| 95.1| 90.38| 98.68| 98.67| 85.7| 88.51| 89.82| 90.3 |
| acc_wang271k_clean_std_of_macbert4mdcspell                             | 85.49| 65.1| 81.54| 87.73| 87.73| 81.4| 90.5| 82.15| 89.4| 92.86| 80.81| 85.19| 86.73| 86.34| 89.6| 95.24 |
| acc_wang271k_clean_std_of_macbert4mdcspell_add_acc_by_true             | 91.74| 76.2| 88.7| 92.36| 92.27| 91.52| 94.2| 90.69| 95.0| 96.06| 87.39| 90.77| 94.19| 91.64| 96.64| 98.46 |
| acc_wang271k_clean_std_of_macbert4mdcspell_add_acc_by_true_mask7_acc15 | 90.3| 73.4| 87.1| 90.82| 90.73| 90.06| 93.0| 88.51| 93.93| 95.17| 85.79| 89.75| 91.98| 90.72| 95.0| 98.52 |
