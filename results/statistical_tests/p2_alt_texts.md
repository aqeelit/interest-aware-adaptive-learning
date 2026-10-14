# Alt texts for Paper 2 figures (factual descriptions of what is plotted)

## F1 (p2_F1_accuracy_by_arm.png / .svg; data: p2_F1_data.csv)

Dot plot in three panels (FULL, EARLY_t28, EARLY_t56). The x-axis lists five arms: base, base_course, W, M, WB. The y-axis shows pooled out-of-fold accuracy (each panel has its own y range). Each panel shows one point per arm for logistic regression (filled circles) and for gradient boosting (open squares).

## F2 (p2_F2_icc_by_feature.png / .svg; data: p2_F2_data.csv)

Horizontal dot plot. The y-axis lists the binary outcome and the 21 interest columns (20 activity-type click shares and resource_diversity_entropy). The x-axis is ICC(1) across the 22 presentations. Three markers per row: FULL (black circle), EARLY_t28 (open triangle) and EARLY_t56 (orange square). Rows where the ICC is undefined because the column is constant are labelled 'undefined' instead of a marker.

## F3 (p2_F3_module_forest.png / .svg; data: p2_F3_data.csv)

Forest plot in four panels (EARLY_t28 and EARLY_t56 by logistic regression and gradient boosting). Each panel lists the seven modules AAA to GGG on the y-axis with the change in accuracy from adding the 21 course-centered interest columns (point and 95% bootstrap interval). A black diamond at the bottom shows the DerSimonian-Laird pooled estimate and its 95% interval, and a dotted line above it shows the Hartung-Knapp interval. A vertical reference line marks zero.

## F4 (p2_F4_early_delta_acc.png / .svg; data: p2_F4_data.csv)

Point-and-interval plot. The x-axis lists four EARLY settings (t28 and t56, each VLE-only and with assessments). The y-axis shows the change in pooled accuracy with its 95% student-level bootstrap interval, with a horizontal reference line at zero. Each setting has four points: logistic regression (circles) and gradient boosting (squares), for base_course to full21_ctr (filled) and for z-scored base_course to the same plus full21_ctr (open).

