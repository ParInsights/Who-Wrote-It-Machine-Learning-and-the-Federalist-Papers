# FEDERALIST PAPERS AUTHORSHIP ATTRIBUTION: PYTHON DECISION TREES AND SUPERVISED CLASSIFIERS

## I. EXECUTIVE SUMMARY

In the late 1780s, nine of the thirteen states still had to approve the new U.S. Constitution.[^1] Alexander Hamilton, James Madison, and John Jay wrote *The Federalist Papers*, a series of 85 letters printed in New York newspapers in 1787 and 1788 urging New York voters to ratify it.[^2] All 85 essays appeared under one pen name, "Publius." Those letters contended that the Constitution would preserve the Union and allow the federal government to act in the nation's interest. They have been regarded ever since as primary evidence of what the framers meant. In fact, in 1821, in *Cohens v. Virginia*, Chief Justice John Marshall called the papers a "complete commentary on our constitution; and is appealed to by all parties in the questions to which that instrument has given birth. Its intrinsic merit entitles it to this high rank, and the part two of its authors performed in framing the constitution, put it very much in their power to explain the views with which it was framed."[^3]

The predicament is that for a document of this caliber in creating and ratifying our Constitution, we still do not know with complete certainty from historical records alone who wrote all 85 essays.[^4] A bound edition was first published in 1788 without author names. Individual names did not appear until printer Jacob Gideon brought out an 1818 edition from Madison's notes.[^5] Hamilton had written a list claiming several of those essays before he was killed in his 1804 duel. Later, Madison also claimed authorship.[^6] Historians were trying ever since to find out which one was the real author.

In the `author` column of our CSV file (`fedPapers85.csv`), there are 74 essays with identified authors: 51 written by Hamilton, 15 by Madison, 3 written jointly by Hamilton and Madison (`HM`), and 5 by Jay.[^7] Eleven essays (Nos. 49–57, 62, and 63) were marked `dispt` for "Hamilton or Madison."[^8] In Method 1 of this project, we drew a conclusion on who wrote those disputed essays with clustering (k-means, expectation-maximization, and hierarchical clustering).[^9] This time, in Method 2, we use decision trees and additional supervised classifiers in Python (`pandas`, `numpy`, `scikit-learn`, and `matplotlib`) to answer that question instead.[^10] We also check which algorithm is the most accurate.[^11]

Our first untuned Python decision tree had 90.0 percent accuracy on the 20-essay test set. However, no papers were assigned to Jay.[^12] Once we raised `max_depth` and pruned the tree with Python's cost-complexity table (`ccp_alpha = 0.030`), Jay was included on a second split using the word "which," and test accuracy went up to 95.0 percent (95.8 percent in five-fold cross-validation).[^13] Even so, a two-word tree sent Federalist No. 51 to Jay because its "which" count was right below the split.[^14] We therefore ran three 70-word models beside the tree: Random Forest (95.9 percent five-fold accuracy), Gradient Boosted Trees (95.8 percent), and L2-regularized logistic regression, which reached 97.2 percent in leave-one-out testing (69 of 71 confirmed papers right).[^15] Ten of the 11 disputed papers went to Madison, and all 3 joint `HM` papers went to Madison too. Federalist No. 55 split two to two.[^16]

## II. FINDINGS AND RECOMMENDATIONS

### A. Findings of Fact

(1) Function-word rates separate Hamilton, Madison, and Jay across the 71 confirmed single-author papers. Hamilton used "upon" at an average rate of 0.0473. Madison's average was 0.0022, and Jay's was 0.0018.[^17] Madison went higher on "on" (0.1051 against Hamilton's 0.0474) and "by" (0.1661 against Hamilton's 0.1046).[^18] In the 11 disputed papers, the averages are 0.0012 for "upon," 0.1125 for "on," and 0.1615 for "by."[^19]

(2) Untuned decision trees leave Jay out. Hamilton wrote 51 of the 71 confirmed single-author essays, while Jay wrote five.[^20] Under default stopping rules (`min_samples_split = 20`) or `max_depth = 1`, the Python tree split once on "upon" <= 0.019 and stopped.[^21] Test accuracy was 90.0 percent, yet zero papers were assigned to Jay.[^22]

(3) Cost-complexity pruning (`ccp_alpha = 0.030`) brings Jay in without overfitting. With zero penalty (`ccp_alpha = 0.000`), Python added a third split on "will" <= 0.105 for one Hamilton paper in the training set. Training accuracy was 100.0 percent. Test accuracy dropped to 85.0 percent.[^23] At `ccp_alpha = 0.030`, the pruned tree kept three leaves on "upon" and "which" and reached 95.0 percent on the test set (19 of 20 right).[^24]

(4) A two-word tree misclassified Federalist No. 51 as Jay. Both our original R tree and our pruned Python tree fit on all 71 confirmed papers assigned 10 disputed essays to Madison and 1 to Jay.[^25] That Jay call was No. 51. Its "which" rate was 0.105, right below the 0.113 cutoff.[^26] When Random Forest and L2 logistic regression scored all 70 words together, No. 51 went to Madison at 61.4 percent and 99.8 percent probability.[^27]

(5) Ten of the 11 disputed papers went to Madison, while Federalist No. 55 split our four models. Nos. 49, 50, 51, 52, 53, 54, 56, 57, 62, and 63 were all assigned to Madison, with logistic regression probabilities from 87.0 percent to 99.8 percent.[^28] No. 55 broke two to two. Our pruned tree and Gradient Boosting picked Madison, while Random Forest (52.8 percent) and logistic regression (80.4 percent) picked Hamilton because "on" (0.051), "by" (0.102), and "to" (0.560) were closer to Hamilton's averages.[^29] Mosteller and Wallace noted the same exception for No. 55 in 1964.[^30]

(6) All three joint `HM` papers (Nos. 18, 19, and 20) matched Madison when tested separately. We held `HM` out of training so shared essays would not skew single-author splits.[^31] In Part 4, Random Forest (50.0 to 76.4 percent) and L2 logistic regression (89.7 to 98.7 percent) assigned all three papers to Madison, which fits historical records that Madison drafted Nos. 18 through 20 from his notes on confederacies with short notes from Hamilton.[^32]

### B. Recommendations

(1) We recommend removing multi-author rows like `HM` before single-author models are trained, and scoring them on their own afterward. When we originally ran our R code and trees with `HM` left in, accuracy was much lower than when we took those three rows out. Once `HM` was separated during cleaning, single-author accuracy went up, and we were still able to score Nos. 18, 19, and 20 in Part 4.

(2) Tree depth should be chosen with Python's `cost_complexity_pruning_path`, and sensitivity should be checked for every author. Our first tree had 90.0 percent overall accuracy on the test set. Even so, Jay's sensitivity was 0.0 percent because zero papers were assigned to him. When author counts are uneven, the confusion matrix has to be checked by class so minority authors are not left out.

(3) A pruned decision tree should be paired with Random Forest and regularized logistic regression. One tree is easy to read and print. Still, two words cannot catch every borderline paper on their own, so we also ran 70-word probability models to check essays like No. 51 and No. 55.

## III. BACKGROUND AND PRIOR WORK

### A. Origins of the Authorship Dispute

Hamilton organized the Federalist project in the fall of 1787 and asked Jay and Madison to help write the series.[^33] All three men used the pen name "Publius" so New York readers would judge the proposed Constitution on its merits.[^34] Jay got sick after writing Nos. 2, 3, 4, and 5. He came back later for just one essay, No. 64.[^35] Hamilton and Madison wrote all the rest under newspaper deadlines, and pages were often at the printer while the next number was still being drafted.[^36]

The fight over authorship started years later. Two days before his July 1804 duel with Aaron Burr, Hamilton left a list in Egbert Benson's law office claiming about two thirds of the 85 essays, including the disputed ones on the House and Senate.[^37] Madison gave printer Jacob Gideon a different list in 1818. Madison claimed Nos. 49 through 58 and Nos. 62 and 63 for himself, and he marked Nos. 18, 19, and 20 as joint work with Hamilton.[^38] Both men wrote in the same eighteenth-century style on the same parts of the Constitution, so historians could not settle the dispute from outside evidence alone.

### B. Mosteller and Wallace and Method 1 Clustering

In the 1960s, statisticians Frederick Mosteller and David L. Wallace analyzed the frequency distributions of common function words in the Federalist Papers and drew their conclusions.[^39] This was pioneering work on mathematical approaches to authorship attribution.[^40] Topic words like *war*, *tax*, or *jury* change from one essay to the next. Function words like "upon," "on," "by," "to," "there," "also," and "which" do not depend on the topic, so they track each writer's habits. Mosteller and Wallace wrote: "On the basis of our data alone, Madison is extremely likely, in the sense of degree of belief, to have written all of the disputed Federalists: Nos. 49 through 58 and 62 and 63, with the possible exception of No. 55."[^41] Nowadays, authorship attribution has become a classic problem in the data mining field, with applications in forensics, deception detection, and information organization.[^42]

In our CSV file (`fedPapers85.csv`), Federalist No. 58 is already labeled `Madison`. That leaves 11 disputed files (Nos. 49–57, 62, and 63).[^43] In Method 1 of this project, we ran k-means, expectation-maximization, and hierarchical clustering on all 85 essays without labels, and those clusters grouped the disputed essays with Madison.[^44] Even so, clustering treats all 70 columns with equal weight and gives no split rules or author probabilities, so in Method 2 we switched to supervised Python classifiers to see what rules are learned and which model is most accurate.

## IV. DATA PREPARATION IN PYTHON

### A. Dataset Structure and Cleaning

Data preparation first required loading the CSV file (`fedPapers85.csv`) in `federalist_authorship_analysis.py` with `pandas.read_csv`, which replaced `file.choose()` and `str(fedpapers)` from our R code.[^45] The table has 85 observations and 72 variables. Column 1 (`author`) has five values: `Hamilton` (51), `Madison` (15), `dispt` (11), `Jay` (5), and `HM` (3). Column 2 (`filename`) gives each file name, and we pulled the essay number from it so our tables sort from 1 to 85.[^46] We then removed `author` and `filename` to leave 70 numeric columns of function-word frequencies. Every cell had a number. Zero values were missing.[^47]

Next, we prepared our data frames for the experiment. In our earlier R script, we created the non-disputed table and split `train` and `test` first. After we checked those early runs, we decided to go back and remove all rows where the author was `HM`. We added `subset(fedpapers_clean, author != "HM")` to the Clean section and reran all lines below it.[^48] We took `HM` out because we did not want our model adjusted for both Madison and Hamilton at the same time. When we originally ran the full R code and trees with `HM` included, accuracy was much lower than when we removed it.

In Python, all three tables were set up right away in Part 1: `known_df` has the 71 confirmed single-author papers, `disputed_df` has the 11 disputed target papers, and `hm_df` holds the 3 joint `HM` papers for Part 4.[^49] Table 1 shows the essay counts and average word rates for each group.

### B. Train-Test Split and Cross-Validation

In R, we split `nondis_df` with `caret::createDataPartition(y = nondis_df$author, p = 0.7, list = FALSE)` so about two thirds went into `train` (51 rows) and one third went into `test` (20 rows).[^50] R printed a warning on that line because `dispt` was still an empty factor level in `author`. Later, we also had issues when we built the test confusion matrix in R. Row names were not lining up, so we had to set `row.names(test) <- NULL`, create a second test frame (`fed_test1`), and change `results` into a factor.[^51]

Python's `scikit-learn` did not have any of those R factor issues. We called `train_test_split(X_known, y_known, test_size=20, stratify=y_known, random_state=3)` and got the same 51-train and 20-test split with exact class shares: 37 Hamilton, 11 Madison, and 3 Jay were in `train`, while 14 Hamilton, 4 Madison, and 2 Jay were in `test`.[^52] A 20-paper test set is small, though. Only two Jay papers and four Madison papers are in `test`, so one wrong prediction changes test accuracy by five percentage points. For that reason, we also ran five-fold stratified cross-validation (`StratifiedKFold`) and 71-fold leave-one-out cross-validation (`LeaveOneOut`) across all 71 confirmed papers.[^53]

## V. DECISION TREE MODELS AND PRUNING

### A. Untuned Baseline Tree and Tuned Model 1

We started in Part 2 by building a general Gini decision tree (`dt_general`) on our 51 training rows with `min_samples_split = 20` and `min_samples_leaf = 7`, which matched R's default `rpart` limits.[^54] Next, we built Tuned Model 1 (`dt_tuned1`) with `min_samples_split = 10` and `max_depth = 1`, just like `Fed_tuned1` in our R code.[^55]

Tuned Model 1 did not change at all from the original untuned tree (Figure 1, Panel A). Both models split once at the root on "upon" <= 0.019 (0.424 root Gini impurity; 14 of 51, or 27.5 percent, root node error).[^56] Thirty-six of Hamilton's 37 training papers had "upon" > 0.019 and went into a pure Hamilton leaf. Fifteen papers (1 Hamilton, 3 Jay, and 11 Madison) went into one Madison leaf.[^57] On the 20-paper test set, this tree got 18 of 20 right, for 90.0 percent test accuracy and 90.2 percent five-fold cross-validation accuracy.[^58] It is key to note that in the original model, no papers were assigned to Jay.[^59] That left node only had 15 rows, so `min_samples_split = 20` (or `max_depth = 1`) stopped the tree before Jay could get a leaf.

### B. Deeper Splits and Cost-Complexity Pruning

Because Tuned Model 1 was identical to the general tree, we changed `max_depth` in our second tuned model (`dt_tuned2`). We kept `min_samples_split` at 10. However, `max_depth` was raised to 5, which was the same setting as `Fed_tuned2` in R.[^60]

In this second tuned model, Jay was included in the decision tree right away. That 15-paper non-Hamilton branch split a second time on "which" <= 0.114 (in R's random split, tied splits picked "and" < 0.624, with "an", "of", and "the" at 1.000 surrogate agreement; on our Python training split, "which" <= 0.114 had the largest Gini drop).[^61] Four training papers (1 Hamilton and 3 Jay) went into a Jay leaf, and all 11 Madison training papers landed in a pure Madison leaf.[^62] Error went down from the first model, holdout test accuracy rose to 95.0 percent (19 of 20 right), and both of Jay's test papers were classified accurately.[^63] We seem to be on the right track.

In our third tuned model (`min_samples_split = 5`, `max_depth = 5`), we lowered `min_samples_split` to 5. Even though the parameters in Model 3 were different from Model 2, the tree did not split any further because the Jay node only held four rows. Both models gave the same three-leaf tree. What if we removed all stopping limits (`ccp_alpha = 0.000`)? Python added a third split on "will" <= 0.105 to pull that single Hamilton paper out of Jay's leaf.[^64] Training accuracy hit 100.0 percent. Holdout test accuracy fell to 85.0 percent (17 of 20).[^65] That third split was overfitting.

In R, our goal during inspection was to find the best tree size with `printcp` and `plotcp`, where we plotted `cp` values until cross-validation error reached its minimum.[^66] In Python, `cost_complexity_pruning_path` builds that same table across `ccp_alpha` (Table 2 and Figure 2). At `ccp_alpha = 0.030`, Python drops the "will" split and keeps the three-leaf tree on "upon" and "which" (Figure 1, Panel B). On the test set, this pruned tree had 95.0 percent accuracy and missed only one Madison paper (Figure 3, Panel A). Across all 71 confirmed papers, it reached 95.8 percent accuracy in both five-fold and leave-one-out cross-validation (68 of 71 right).[^67]

## VI. ENSEMBLE AND PROBABILITY CLASSIFIERS

### A. Limits of a Single Decision Tree

Even after pruning, one decision tree only checks two words ("upon" and "which") out of 70.[^68] We leave the other 68 columns unused whenever one tree predicts a paper. Suppose Madison wrote an essay that matched his usual rates on "upon," "on," "by," "there," "to," and "would," yet used "which" just below 0.113 (the cutoff when the pruned tree is fit on all 71 confirmed papers). A single tree has nowhere else to send that file, so it assigns Jay. That is why our R decision tree assigned 10 disputed papers to Madison and 1 to Jay.[^69]

### B. Random Forest, Gradient Boosting, and Logistic Regression

In our README, we also asked a second question: which algorithms give the most accurate results?[^70] We tested three multi-word models in Part 3 of `federalist_authorship_analysis.py`.

First, we built a 500-tree Random Forest (`max_features = "sqrt"`, `class_weight = "balanced"`). Each tree draws a bootstrap sample of training papers and checks 8 random words at every split, so whenever "upon" or "which" is left out of those 8 words, the tree has to learn from other function words in the table.[^71] Test accuracy was 95.0 percent, and five-fold cross-validation accuracy was 95.9 percent. In the Gini importance chart (Figure 4, Panel A), "upon" came in first, followed by "of," "and," "an," "which," "the," "there," "to," "on," "would," "also," and "by."[^72]

Next, we ran Gradient Boosted Decision Trees (200 trees, `learning_rate = 0.05`, `max_depth = 2`). Instead of averaging independent trees, Gradient Boosting builds shallow two-level trees one after another so each tree fixes mistakes made by earlier trees.[^73] Test accuracy was 95.0 percent. Five-fold and leave-one-out cross-validation both came in at 95.8 percent.[^74]

Finally, we fit L2-regularized multinomial logistic regression (`StandardScaler` with `LogisticRegression(C = 0.5, class_weight = "balanced")`). Logistic regression weights all 70 standardized word columns at the same time instead of stopping after two hard cutoffs.[^75] It was our most accurate model in cross-validation: 97.2 percent in five-fold cross-validation and 97.2 percent in leave-one-out testing, where 69 of the 71 confirmed papers were right (Table 3 and Figure 3, Panel B).[^76]

## VII. DISPUTED AND JOINT ESSAY PREDICTIONS

### A. Predictions on the Eleven Disputed Papers

Now that our models were tested, we refit the four main classifiers on all 71 confirmed single-author papers and ran predictions on the 11 rows in `disputed_df` (Table 4 and Figure 5).[^77] Hamilton had none of the disputed papers assigned by unanimous consensus. Ten of the 11 disputed papers had "upon" = 0.000, and No. 50 was 0.013, well below Hamilton's 0.047 average (Figure 4, Panel B).[^78] On nine disputed papers (Nos. 49, 50, 52, 53, 54, 56, 57, 62, and 63), all four models picked Madison, with logistic regression probabilities from 87.0 percent to 99.3 percent.[^79]

Federalist No. 51 explained the single Jay call from our earlier R report. In R, our prediction results showed that out of the 11 disputed papers, 10 were written by Madison and 1 was written by Jay.[^80] Our pruned Python tree gave that exact same count, and Table 4 shows that the Jay paper was No. 51. No. 51 had "upon" = 0.000 and a high Madison "on" rate of 0.150, yet its "which" rate was 0.105, right below the 0.113 split.[^81] Jay was ill during that stretch of essays and never claimed No. 51. Once Random Forest and L2 logistic regression weighed all 70 words, both assigned No. 51 to Madison (61.4 percent in Random Forest and 99.8 percent in logistic regression).[^82]

Federalist No. 55 was the only disputed essay that split our models.[^83] The pruned tree and Gradient Boosting assigned No. 55 to Madison ("upon" = 0.000, "which" = 0.211). Random Forest (52.8 percent Hamilton versus 36.6 percent Madison) and L2 logistic regression (80.4 percent Hamilton versus 18.3 percent Madison) assigned it to Hamilton.[^84] In Table 4, we can see why: No. 55 had "on" = 0.051 (right beside Hamilton's 0.047 mean instead of Madison's 0.105), "by" = 0.102 (Hamilton averaged 0.105; Madison averaged 0.166), and "to" = 0.560.[^85] Mosteller and Wallace flagged that same essay in 1964 when they gave the disputed group to Madison "with the possible exception of No. 55."[^86]

### B. Joint `HM` Essays (Nos. 18, 19, and 20)

One of the main questions raised in our R conclusion was what should be done with the papers assigned jointly to Hamilton and Madison (`HM`: Nos. 18, 19, and 20).[^87] We had removed `HM` from training because we were worried joint authorship would confuse single-author trees. At the time, we noted that further analysis was needed on those three essays.

In Part 4 of `federalist_authorship_analysis.py`, we scored `hm_df` across all four trained models. The single pruned tree sent No. 18 to Jay on a low "which" rate (0.081) and sent Nos. 19 and 20 to Madison. Random Forest (50.0, 76.4, and 57.6 percent Madison) and L2 logistic regression (98.7, 95.4, and 89.7 percent Madison) assigned all three `HM` papers to Madison (Table 4 and Figure 5).[^88] That result lines up with historical accounts showing Madison drafted Nos. 18 through 20 from his notes on ancient and modern confederacies and folded in short notes from Hamilton.[^89]

## VIII. CONCLUSION

While *The Federalist Papers* remain central to the history and ratification of the U.S. Constitution, who wrote the 11 disputed papers was long treated as an open mystery.[^90] When we rebuilt our R decision tree code in Python, cost-complexity pruning (`ccp_alpha = 0.030`) gave a three-leaf tree on "upon" and "which" with 95.0 percent test accuracy and 95.8 percent cross-validation accuracy, and L2-regularized logistic regression raised leave-one-out accuracy across all 71 confirmed papers to 97.2 percent.[^91] Madison wrote 10 of the 11 disputed papers (Nos. 49–54, 56–57, 62, and 63) according to our models, and all three joint `HM` papers (Nos. 18–20) matched Madison's word rates as well. Federalist No. 55 was the only exception. Two models picked Madison, and two picked Hamilton.[^92]

[^1]: See Parin Patel, *Use Decision Trees to Solve a Mystery in History* (Method 2 R Report), p. 1; Library of Congress, *Federalist Papers: Primary Documents in American History*, https://guides.loc.gov/federalist-papers/full-text.
[^2]: See ParInsights, *Who Wrote It: Machine Learning and the Federalist Papers* (`README.md`), https://github.com/ParInsights/Who-Wrote-It-Machine-Learning-and-the-Federalist-Papers/blob/master/README.md.
[^3]: *Cohens v. Virginia*, 19 U.S. (6 Wheat.) 264, 418 (1821), https://supreme.justia.com/cases/federal/us/19/264/.
[^4]: See Parin Patel, Method 2 R Report, supra note 1, p. 1.
[^5]: See ParInsights, `README.md`, supra note 2 ("1. About the Federalist Papers").
[^6]: Id. ("2. About the disputed authorship").
[^7]: Id.; verified in `fedPapers85.csv` (85 rows x 72 columns) via `federalist_authorship_analysis.py` (Appendix A, Part 1).
[^8]: Verified in `fedPapers85.csv` (`dispt_fed_49.txt` through `dispt_fed_57.txt`, `dispt_fed_62.txt`, and `dispt_fed_63.txt`).
[^9]: See ParInsights, `README.md`, supra note 2 ("Method 1 involves using clustering algorithms - k-Means, EM, and HAC.").
[^10]: See ParInsights, *Method 2 R Code - Using a Decision Tree.R*, https://github.com/ParInsights/Who-Wrote-It-Machine-Learning-and-the-Federalist-Papers/blob/master/Method%202%20R%20Code%20-%20Using%20a%20Decision%20Tree%20.R; Appendix A (`federalist_authorship_analysis.py`).
[^11]: See ParInsights, `README.md`, supra note 2 ("We will also try to see which algorithms yield the most accurate results?").
[^12]: See Appendix A, Part 2 and Table 3 (`DT Baseline` and `DT Tuned 1`: 90.0% test accuracy, 0.0% Jay sensitivity).
[^13]: Id., Part 2 and Table 2 (`DT Pruned` at `ccp_alpha = 0.030`: 95.0% test accuracy, 95.81% 5-fold CV).
[^14]: Id., Part 4 and Table 4 (Federalist No. 51 `"which" = 0.105` vs. `0.113` full-corpus split).
[^15]: Id., Part 3 and Table 3 (`RandomForestClassifier`, `GradientBoostingClassifier`, and `LogisticRegression` benchmarks).
[^16]: Id., Part 4, Table 4, and Figure 5.
[^17]: Id., Part 1 and Table 1 (group means computed across `fedPapers85.csv`).
[^18]: Id. (`"on"` and `"by"` means for Hamilton `n = 51` and Madison `n = 15`).
[^19]: Id. (`dispt` cohort `n = 11` means for `"upon"`, `"on"`, and `"by"`).
[^20]: Id., Part 1 (`known_df` counts: 51 Hamilton, 15 Madison, 5 Jay).
[^21]: Id., Part 2 (`dt_general` and `dt_tuned1` root split on `"upon" <= 0.019`).
[^22]: Id., Table 3 and Figure 1, Panel A.
[^23]: Id., Table 2 (`ccp_alpha = 0.00000`, 4 leaves, train = 100.0%, test = 85.0%).
[^24]: Id. (`ccp_alpha = 0.030`, 3 leaves, train = 98.0%, test = 95.0%).
[^25]: Compare Parin Patel, Method 2 R Report, supra note 1, p. 12, with Table 4 (`DT Pruned` column).
[^26]: See Table 4 (Federalist No. 51 word rates).
[^27]: Id. (Federalist No. 51: `rf_madison_prob = 0.614`, `lr_madison_prob = 0.998`).
[^28]: Id. and Figure 5.
[^29]: Id. (Federalist No. 55: `"upon" = 0.000`, `"on" = 0.051`, `"which" = 0.211`, `"by" = 0.102`, `"to" = 0.560`).
[^30]: See Frederick Mosteller and David L. Wallace, *Inference and Disputed Authorship: The Federalist* (Reading, MA: Addison-Wesley, 1964), https://casbs.stanford.edu/inference-and-disputed-authorship-federalist.
[^31]: See Appendix A, Part 1 (`hm_df`) and Parin Patel, Method 2 R Report, supra note 1, pp. 11–12.
[^32]: See Table 4 (`hm_predictions` for Nos. 18–20); Library of Congress, *Federalist Nos. 11–20*, https://guides.loc.gov/federalist-papers/text-11-20.
[^33]: See Library of Congress, *About the Federalist Papers*, https://guides.loc.gov/federalist-papers/full-text.
[^34]: Id.
[^35]: Id. (Jay authored Nos. 2, 3, 4, 5, and 64).
[^36]: See Mosteller and Wallace, supra note 30.
[^37]: Id.; National Archives, *Founders Online: Alexander Hamilton's List of Authors*, https://founders.archives.gov/.
[^38]: See ParInsights, `README.md`, supra note 2 (discussing the 1818 Jacob Gideon edition).
[^39]: Id. ("3. Computational approach for authorship attribution").
[^40]: Id.; CMU Stat 724 Summary, http://www.stat.cmu.edu/~vlachos/courses/724/final/mosteller.pdf.
[^41]: Mosteller and Wallace, supra note 30, quoted at https://thirty-thousand.org/supplemental/federalist55_authorship/.
[^42]: See ParInsights, `README.md`, supra note 2.
[^43]: Verified in `fedPapers85.csv` (`Madison_fed_58.txt` carries `author = Madison`).
[^44]: See ParInsights, `README.md`, supra note 2 ("Method 1 involves using clustering algorithms - k-Means, EM, and HAC.").
[^45]: Compare Parin Patel, Method 2 R Report, supra note 1, pp. 1–2, with Appendix A, Part 1.
[^46]: See Appendix A, Part 1 (`essay_number` function).
[^47]: Id. (zero missing values across 85 x 70 numeric entries).
[^48]: See Parin Patel, Method 2 R Report, supra note 1, pp. 2–3, 11.
[^49]: See Appendix A, Part 1 (`known_df`, `disputed_df`, `hm_df`).
[^50]: See Parin Patel, Method 2 R Report, supra note 1, p. 2, Figure 4.
[^51]: Id., pp. 9–10, Figure 22 (`row.names(test) <- NULL` and `fed_test1`).
[^52]: See Appendix A, Part 1 (`train_test_split` with `test_size=20, stratify=y_known, random_state=3`).
[^53]: Id. (`StratifiedKFold` and `LeaveOneOut`).
[^54]: Id., Part 2 (`dt_general`).
[^55]: Compare ParInsights, *Method 2 R Code*, supra note 10 (`Fed_tuned1`), with Appendix A, Part 2 (`dt_tuned1`).
[^56]: See Figure 1, Panel A and Parin Patel, Method 2 R Report, supra note 1, pp. 4–6, 11.
[^57]: See Figure 1, Panel A.
[^58]: See Table 3 (`DT Baseline` and `DT Tuned 1`).
[^59]: See Parin Patel, Method 2 R Report, supra note 1, p. 11 ("It is key to note that in the original model, no papers have been assigned to Jay.").
[^60]: Compare ParInsights, *Method 2 R Code*, supra note 10 (`Fed_tuned2`), with Appendix A, Part 2 (`dt_tuned2`).
[^61]: Compare Parin Patel, Method 2 R Report, supra note 1, pp. 7, 12, with Figure 1, Panel B.
[^62]: See Figure 1, Panel B.
[^63]: See Table 3 and Figure 3, Panel A.
[^64]: See Table 2 (`ccp_alpha = 0.00000`, 4 leaves).
[^65]: Id. (train = 100.0%, holdout test = 85.0%).
[^66]: See Parin Patel, Method 2 R Report, supra note 1, pp. 4–5, 12 (`printcp` and `plotcp`).
[^67]: See Table 2, Table 3, Figure 2, and Figure 3.
[^68]: See Appendix A, Part 4 (`dt_pruned_rules`).
[^69]: See Parin Patel, Method 2 R Report, supra note 1, p. 12.
[^70]: See ParInsights, `README.md`, supra note 2; Appendix A, Part 3.
[^71]: See Leo Breiman, "Random Forests," *Machine Learning* 45, no. 1 (2001): 5–32, https://link.springer.com/article/10.1023/A:1010933404324.
[^72]: See Figure 4, Panel A and `outputs/analysis_results.json` (`top12_rf_features`).
[^73]: See scikit-learn documentation, `GradientBoostingClassifier`, https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html.
[^74]: See Table 3 (`Gradient Boosted Trees`).
[^75]: See scikit-learn documentation, `LogisticRegression`, https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html.
[^76]: See Table 3 and Figure 3, Panel B (`L2 Logistic Regression`: 97.2% 5-fold CV, 97.2% LOOCV).
[^77]: See Appendix A, Part 4 (`attribute_cohort(disputed_df)`).
[^78]: See Table 4 and Figure 4, Panel B.
[^79]: See Table 4 (`Nos. 49, 50, 52, 53, 54, 56, 57, 62, 63`).
[^80]: See Parin Patel, Method 2 R Report, supra note 1, p. 12.
[^81]: See Table 4 (Federalist No. 51: `"upon" = 0.000`, `"on" = 0.150`, `"which" = 0.105`).
[^82]: Id. (`rf_madison_prob = 0.614`, `lr_madison_prob = 0.998`).
[^83]: Id. (`essay_no: 55`, `Madison (Split 2-2)`).
[^84]: Id. (`outputs/analysis_results.json`, `essay_no: 55`).
[^85]: Id. (Federalist No. 55 word rates vs. Table 1 group means).
[^86]: See Mosteller and Wallace, supra note 30; Jacques Savoy, "Authorship of the Disputed Federalist Papers with Burrows' Delta" (2013), https://pdfs.semanticscholar.org/986a/37fa5110ce624df2cc26aa077b8440444a55.pdf.
[^87]: See Parin Patel, Method 2 R Report, supra note 1, p. 12 ("One of the main questions raised through analysis is what should be done with papers assigned to Hamilton and Madison 'HM'.").
[^88]: See Table 4 and Figure 5 (`hm_predictions`).
[^89]: See Library of Congress, *Federalist Nos. 11–20*, supra note 32.
[^90]: See Parin Patel, Method 2 R Report, supra note 1, p. 12.
[^91]: See Tables 2 and 3; Appendix A (`federalist_authorship_analysis.py`).
[^92]: See Table 4 and Figure 5.
