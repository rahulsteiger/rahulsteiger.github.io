---
layout: distill
title: Ensemble Methods
description: Ensemble methods combine multiple simple learning algorithms to achieve superior overall performance. This blog post is an adaptation of a group project from the CS4270 course I took at the National University of Singapore during my exchange.

tags: Decision-Trees Random-Forest Boosting XGBoost 
giscus_comments: true
date: 2025-01-26
featured: false
mermaid:
  enabled: true
  zoomable: true
code_diff: true
map: true
chart:
  chartjs: true
  echarts: true
  vega_lite: true
tikzjax: true
typograms: true

authors:
  - name: J.S
    affiliations:
      name: NUS, ETH Zurich
  - name: Rahul Steiger
    affiliations:
      name: NUS, ETH Zurich


bibliography: 2025-01-26-ensemble-methods.bib

# Optionally, you can add a table of contents to your post.
# NOTES:
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
#   - we may want to automate TOC generation in the future using
#     jekyll-toc plugin (https://github.com/toshimaru/jekyll-toc).
toc:
  - name: Introduction
    # if a section has subsections, you can add them as follows:
    # subsections:
    #   - name: Example Child Subsection 1
    #   - name: Example Child Subsection 2
  - name: Decision Trees
  - name: Random Forest
  - name: Extremely Randomized Trees
  - name: Gradient Boosting
  - name: XGBoost
  - name: Experiment
  - name: Appendix

# Below is an example of injecting additional post-specific styles.
# If you use this post as a template, delete this _styles block.
_styles: >
  .fake-img {
    background: #bbb;
    border: 1px solid rgba(0, 0, 0, 0.1);
    box-shadow: 0 0px 4px rgba(0, 0, 0, 0.1);
    margin-bottom: 12px;
  }
  .fake-img p {
    font-family: monospace;
    color: white;
    text-align: left;
    margin: 12px 0;
    text-align: center;
    font-size: 16px;
  }
---

## Introduction

Ensemble methods combine multiple simple learning algorithms to produce an overall better result. First, we will provide a brief overview of weak and strong learners and the concept of a decision tree. In the next step, we will introduce two ensemble methods based on bagging: random forests and extremely randomized trees. The second part of the blog post will detail gradient boosting algorithms, focusing on introducing the conventional Gradient Boosting Machine and discussing the principle ideas behind the notoriously popular method XGBoost. Finally, we will conclude with a comparative study of the performance of these models on a representative sample dataset. 


### Weak vs Strong classifiers

A so-called weak learner or base learner is an algorithm or model that performs "slightly better than chance" <d-cite key="murphy-2012" page="555"></d-cite>. This means that the model has some but minimal predictive power. This base learner could, for example, be a shallow decision tree such as a decision tree stump explored in [this section](#decision-trees). A strong learner is a model that can have an arbitrarily small error in the training data set <d-cite key="murphy-2012"></d-cite>.

<strong> $\gamma$-Weak Learnability <d-cite key="shwartz-2014"></d-cite><d-cite key="princeton-weak-learnability"></d-cite></strong>
<p>A learning problem is defined as $\gamma$-weakly learnable for a given hypothesis class $\mathcal{H}$ if for $\gamma > 0$ there exists a function $m_\mathcal{H} : (\delta, \gamma) \to \mathbb{N}$ and an effective algorithm such that for any $0 < \delta < 1$ when the algorithm processes $m_\mathcal{H}(\delta, \gamma)$ examples drawn from a distribution $\mathcal{D}$ over the input-output space $\mathcal{X} \times \mathcal{Y}$, it produces a hypothesis $h \in \mathcal{H}$ that achieves the following error rate with probability of least $1 - \delta$:</p>
<p>
$$
\mathrm{error}_{\mathcal{D}}(h) \leq \frac{1}{2} - \gamma.
$$
</p>

<p>Strong learnability can be defined completely analogously using the error bound $\mathrm{error}_{\mathcal{D}}(h) \leq \gamma$.</p>


#### Combining weak regression learners to reduce variance

Consider a standard regression problem with a dataset: $\mathcal{D} = \\{ (x_i, y_i) \mid i = 1, \dots, \lvert \mathcal{D} \rvert \\}$. 
Assume we are given a set of $n$ regression estimators: $\hat{f}_1(x), \dots, \hat{f}_n(x)$. We define:

$$
\hat{f}(x) = \frac{1}{n} \sum_{i=1}^n \hat{f}_i(x)
$$

We can write the bias-variance decomposition of $\hat{f}$ as follows:

$$
\begin{align*}
\mathbb{E}[(\hat{f}(x) - \mathbb{E}[y \mid x])^2] 
&= (\mathbb{E}[\hat{f}(x)] - \mathbb{E}[y \mid x])^2 + \mathbb{E}[(\hat{f}(x) - \mathbb{E}[\hat{f}(x)])^2] \\
&= \operatorname{Bias}[\hat{f}(x)]^2 + \operatorname{Var}[\hat{f}(x)]
\end{align*}
$$

By the definition of $\hat{f}$, the bias term simplifies to:

$$
\begin{align*}
\operatorname{Bias}[\hat{f}(x)] 
&= \mathbb{E}[\hat{f}(x)] - \mathbb{E}[y|x] \\
&= \frac{1}{n} \sum_{i=1}^n \mathbb{E}[\hat{f}_i(x)] - \mathbb{E}[y \mid x] \\
&= \frac{1}{n} \sum_{i=1}^n [\mathbb{E}[\hat{f}_i(x)] - \mathbb{E}[y \mid x]] \\
&= \frac{1}{n} \sum_{i=1}^n \operatorname{Bias}[\hat{f}_i(x)]
\end{align*}
$$

By the definition of $\hat{f}$, the variance term can be rewritten as:

$$
\begin{align*}
    \operatorname{Var}[\hat{f}(x)] 
    &= \mathbb{E}[(\hat{f}(x) - \mathbb{E}[\hat{f}(x)])^2] \\
    &= \mathbb{E}\left[\left(\frac{1}{n} \sum_{i=1}^n \hat{f}_i(x) - \frac{1}{n} \sum_{i=1}^n \mathbb{E}[\hat{f}_i(x)]\right)^2\right] \\
    &= \mathbb{E}\left[\left(\frac{1}{n}\sum_{i=1}^n \left(\hat{f}_i(x) - \mathbb{E}[\hat{f}_i(x)]\right)\right)^2\right] \\
    &= \frac{1}{n^2} \sum_{i=1}^n \sum_{j=1}^n  \mathbb{E}\left[\left(\hat{f}_i(x) - \mathbb{E}[\hat{f}_i(x)]\right) \left(\hat{f}_j(x) - \mathbb{E}[\hat{f}_j(x)]\right)\right] \\
    &= \frac{1}{n^2} \sum_{i=1}^n \operatorname{Var}[\hat{f}_i(x)] + \frac{1}{n^2} \sum_{i \neq j} \operatorname{Cov}[\hat{f}_i(x), \hat{f}_j(x)]  
\end{align*}
$$

Assuming that $f_i(x)$ and $f_j(x)$ are independent of each other:

$$
\operatorname{Cov}[\hat{f}_i(x), \hat{f}_j(x)] = 0 \quad \text{for } i \neq j,
$$

we will get that:

$$
\operatorname{Var}[\hat{f}(x)] = \frac{1}{n^2} \sum_{i=1}^n \operatorname{Var}[\hat{f}_i(x)].
$$

Consequently, we can reduce the variance arbitrarily by just using more estimators. However, assuming that $f_1, \dots, f_n$ are mutually independent is unrealistic if we assume they are trained on the same dataset. In this blog post, we will discuss algorithms that combine multiple weak learners while ensuring that $\operatorname{Cov}[\hat{f}_i(x), \hat{f}_j(x)] \approx 0$ and a sufficiently low bias.

---

## Decision Trees

Decision trees are a popular sequential model, relying on the recursive partition of the input space into disjoint subspaces and then training a model in each resulting region. The decision tree splits the space by applying a test to check if a specific value fulfills a certain splitting condition. Decision trees can be interpreted as a predictor $h : \mathcal{X} \to \mathcal{Y}$ where the prediction is based on the traversal of the tree from the root to a leaf. One of their main advantages is their natural interpretation. A decision tree can be used for both classification and regression purposes. However, we will focus on the former in this section. <d-cite key="murphy-2012"></d-cite> <d-cite key="shwartz-2014"></d-cite> <d-cite key="kotsiantis-2013"></d-cite>


### Main Idea

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/ensemble-methods/Decision_Tree_v2.jpg" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

*Figure 1: Schematic representation of partitioning of the feature space into two disjoint subspaces based on feature ${x}_i$ and threshold $\theta$. The space for ${x}_i < \theta$ is patterned in red. Each side corresponds to exactly one leaf.*

Following the discussion in <d-cite key="shwartz-2014"></d-cite>, one way to split the data is by applying a threshold $\theta$ to a given numeric feature dimension. We can check if the $i$'th feature of $\mathbf{x}$ is smaller than the threshold: $x_i < \theta$. Therefore, we move to the left child if $\mathbb{1}_{[\mathbf{x}_i < \theta]}$ as illustrated in Figure 1. This then divides our $d$-dimensional space $\mathcal{X}=\mathbb{R}^d$ into two parts with a direct correspondence between the leaves and subspaces. In general, the decision tree divides the feature space into axis-aligned hyperplanes <d-cite key="rokach-2016"></d-cite>. It is also possible to follow a similar procedure for categorical values and check them against a set of values <d-cite key="kotsiantis-2013"></d-cite>.

### Growth Stage

**Algorithm ID3** for $\{-1, 1\}^d \to \{-1, 1\}$, *adapted from <d-cite key="shwartz-2014"></d-cite>, <d-cite key="kaewrod-2018" fig="1"></d-cite>*
$$
\begin{array}{ll}
\textbf{Input:} & \text{Training set } S \subseteq \mathcal{D} = \{(\mathbf{x}_i, y_i) \mid \mathbf{x}_i \in \{-1, 1\}^d, y_i \in \{-1, 1\}\}_{i=1}^n, \\
                & \text{Subset of features } F \subseteq \{1, \ldots, d\} \\
\textbf{Output:} & \text{Decision tree}
\end{array}
$$

$$
\begin{array}{l}
\textbf{function} \texttt{ID3}(S, F) \\
\quad \textbf{if } F = \emptyset \textbf{ or all } y_i \text{ in } S \text{ are the same} \\
\quad \quad \textbf{return } \operatorname{Leaf}(\operatorname{mode}(Y)) \text{ with } Y = \{y_1, \ldots, y_n\} \\
\quad \textbf{end if} \\
\quad \text{Select feature } j \text{ maximizing information gain: } j = \underset{i \in F}{\arg \max} \operatorname{InfoGain}(S, i) \\
\quad \text{Partition } S \text{ into } S_{+1} = \{(\mathbf{x}, y) \in S \mid x_j = +1\} \text{ and } S_{-1} = \{(\mathbf{x}, y) \in S \mid x_j = -1\} \\
\quad \textbf{if } S_{+1} = \emptyset \textbf{ or } S_{-1} = \emptyset \\
\quad \quad \textbf{return } \operatorname{Leaf}(\operatorname{mode}(Y)) \\
\quad \textbf{else} \\
\quad \quad \text{Left branch } T_{+1} \leftarrow \texttt{ID3}(S_{+1}, F \setminus \{j\}) \\
\quad \quad \text{Right branch } T_{-1} \leftarrow \texttt{ID3}(S_{-1}, F \setminus \{j\}) \\
\quad \quad \textbf{return } \text{tree with decision node on } x_j, \text{ left branch } T_{-1}, \text{ and right branch } T_{+1} \\
\quad \textbf{end if} \\
\textbf{end function}
\end{array}
$$

Constructing an optimal decision tree---i.e., a decision tree optimal w.r.t. some gain metric---is generally an NP-hard problem and thus computationally very costly to solve. However, several greedy algorithms that are locally optimal have been developed that can solve the problem approximately but with reasonable efficiency in practical cases. <d-cite key="shwartz-2014"></d-cite> <d-cite key="kotsiantis-2013"></d-cite>

We start by assigning the root node the majority class label and then build the decision tree by iteratively finding the best split in our input data, i.e., we partition the feature space into two or more parts. Following a greedy approach, we only consider one leaf at each iteration and try to maximize some gain metric (later defined) by splitting based on one feature. After computing the gain for all possible splits, we select the split with the highest gain. A leaf is defined as pure if only one class label is associated with it; otherwise, it is considered impure. We grow the tree until the tree either consists only of pure leaves or the gain metric falls below a certain threshold. It is important to store the class labels for each leaf, as these are not necessarily pure. <d-cite key="murphy-2012"></d-cite> <d-cite key="shwartz-2014"></d-cite> <d-cite key="kotsiantis-2013"></d-cite>

### Gain Metric: Information Gain

Several gain metrics are used in practice, and some metrics may lead to trees with high variance but low bias, while others might produce the opposite effect. The most commonly used metrics are information gain and Gini impurity <d-cite key="kotsiantis-2013"></d-cite>. Let us focus on the definition of the information gain.

Information gain is a measure used to determine the effectiveness of a feature in splitting a dataset into subgroups that are more homogeneous in terms of the output variable. Consider a dataset $$\mathcal{D} = \\{(\mathbf{x}_i, y_i)\\}_{i=1}^n$$ consisting of pairs of feature vectors $\mathbf{x}_i$ and their corresponding labels $y_i$. For a subset $S \subseteq \mathcal{D}$, we assess the potential of the $i$-th feature of $\mathbf{x}$, denoted as $x_i$, to split $S$ into smaller subsets. The set of all possible values $x_i$ is represented by $\operatorname{values}(i)$. For each value $v$ that $x_i$ might assume, we define $S_i(v) \subseteq S$ as the subset of $S$ where the $i$-th feature's value is $v$: $$S_i(v) = \\{\mathbf{x} \in S \mid x_i = v\\}$$.

The conditional entropy of $S$ given the feature $x_i$, denoted as $\operatorname{H}(S \mid i)$, quantifies the entropy within $S$ after it has been partitioned according to $x_i$'s values. It is calculated as follows:

The conditional entropy $\operatorname{H}(S \mid i)$ is defined as:

$$
\operatorname{H}(S \mid i) = \sum_{v \in \operatorname{values}(i)} \frac{\mid S_i(v) \mid}{\mid S \mid} \cdot \operatorname{H}\left( S_i(v) \right),
$$

where $\operatorname{H}\left( S_i(v) \right)$ represents the Shannon entropy of subset $S_i(v)$, computed using the formula:

$$
H = - \sum_k p_k \log{p_k},
$$

with $p_k$ denoting the relative frequency of class $k$ within the subset.

The information gain from splitting $S$ on the feature $x_i$, denoted as $\operatorname{InfoGain}(S, i)$, is then defined as the reduction in entropy achieved by this partitioning. It is computed as the difference between the original entropy of $S$ and the weighted average entropy after the split based on $x_i$ <d-cite key="kotsiantis-2013"></d-cite> <d-cite key="myles-2004"></d-cite>:

$$
\begin{align*}
\operatorname{InfoGain}(S, i) 
&= \operatorname{H(S)} - \operatorname{H}(S \mid i) \\
&= \operatorname{H(S)} - \sum_{v \in \operatorname{values}(i)} \frac{\mid S_i(v) \mid}{\mid S \mid} \cdot \operatorname {H}\left( S_i(v) \right),
\end{align*}
$$

where $\operatorname{H(S)}$ is the Shannon entropy of the entire subset $S$ before the split. This measure serves as a criterion for selecting the feature that best divides the dataset into groups with more distinct outcomes, thereby enabling the construction of decision trees where, for every leaf, we would choose the split that maximizes the information gain or equivalently minimizes the entropy <d-cite key="murphy-2012"></d-cite>.

### Pruning Stage

The general problem for decision trees is that they tend to overfit if fully grown, resulting in very little training loss. On the contrary, if the tree is shallow, it generalizes better but will have a higher bias. One way to resolve this problem is by selectively removing parts of the tree. <d-cite key="shwartz-2014"></d-cite>

Pruning, i.e., removing parts of the tree, improves its generalization to unseen data and thus tries to prevent overfitting. We differentiate between pre- and post-pruning, where we either impose a stopping criterion while building the tree or after the tree has been fully built <d-cite key="murphy-2012"></d-cite>.

For pre-pruning, during the growth of the decision tree, certain criteria can be imposed, e.g., the maximum depth of the tree or the minimum information gain of a split. If these criteria can no longer be met, the algorithm terminates. This is referred to as pre-pruning or early-stopping. <d-cite key="kotsiantis-2013"></d-cite>

Post-pruning allows the tree first to grow fully. It then undergoes a pruning process where nodes are evaluated for removal based on their contribution to the model's classification accuracy on a validation set. One example is the so-called "minimal-cost complexity pruning" that involves balancing the accuracy of the tree against its complexity <d-cite key="breiman-1984"></d-cite> referenced in <d-cite key="murphy-2012"></d-cite>. In essence, minimal cost-complexity pruning is about finding a tree that is complex enough to model the underlying patterns in the data accurately but not so complex that it memorizes the training data and performs poorly on unseen data. <d-cite key="murphy-2012"></d-cite> <d-cite key="shwartz-2014"></d-cite> <d-cite key="kotsiantis-2013"></d-cite> <d-cite key="breiman-1984"></d-cite>

## Random Forest

Deep decision trees are prone to overfitting, limiting their generalization ability to unseen data. One improvement is the implementation of Random Forests, a concept pioneered by <d-cite key="breiman-2001"></d-cite>, which is widely popular with over 58,000 citations according to Web of Science/Inspec®.

The main idea is to grow and combine an ensemble of many (potentially shallow) decision trees with some "injected randomness" to reduce bias on unseen data <d-cite key="breiman-2001"></d-cite>. This method significantly reduces the overfitting tendency of individual trees, thereby enhancing the model's predictive performance on new data. Generally, random forests tend to have high predictive accuracy <d-cite key="murphy-2012"></d-cite>.

**Random Forest Training Algorithm**, adapted from <d-cite key="shwartz-2014"></d-cite>, <d-cite key="weinberger-2018"></d-cite>
$$
\\
\begin{array}{l}
\textbf{Input:} & \text{Data Set } \mathcal{D} = \{(\mathbf{x}_i, y_i) \mid \mathbf{x}_i \in \mathbb{R}^d \}_{i=1}^n \\
\textbf{Output:} & \text{Classifier } \hat{h}(\mathbf{x}) = \frac{1}{M} \sum_{j=1}^{M} h_j(\mathbf{x})
\end{array}
$$
$$
\begin{array}{l}
\textbf{Procedure: } \text{Random-Forest(S)} \\
\quad \textbf{Initialize: } \text{Ensemble of trees } \mathcal{H} = \emptyset \\
\quad \text{Sample } M \text{ datasets } \{S_j\}_{j=1}^M \text{ from } \mathcal{D} \text{ with replacement, each of size } m \leq n \\
\quad \textbf{for } j = 1 \text{ to } M \textbf{ do} \\
\quad \quad \text{Sample a subset of features } I_t \subseteq [d] \text{ of size } \mid I_t \mid = k \leq d \text{ without replacement} \\
\quad \quad \text{Train a decision tree } h_j \text{ on dataset } S_j \text{ using only features from } I_t \text{ at each split } t \\
\quad \quad \text{Add } h_j \text{ to the ensemble } \mathcal{H} \leftarrow \mathcal{H} \cup \{h_j\} \\
\quad \textbf{end for} \\
\quad \textbf{Return: } \text{Classifier } \hat{h}(\mathbf{x}) = \frac{1}{M} \sum_{j=1}^{M} h_j(\mathbf{x})
\end{array}
$$

A Random Forest builds upon a dataset $\mathcal{D} = \\{ (\mathbf{x}_i, y_i) \mid \mathbf{x}_i \in \mathbb{R}^d \\}\_{i=1}^n$ by generating $M$ decision trees, each from a bootstrap sample $S_j$ of $S = \\{\mathbf{x} \in \mathcal{D} \\}$. A bootstrap sample is a multiset of $S$ created by sampling with replacement following a uniform distribution over $S$.

In contrast to the conventional decision tree algorithm, the random forest only considers the best split among $k$ features. At each split during the tree-building process, these $k$ features are uniformly sampled without replacement from the $d$ total features. This introduces variability and enhances the model's generalization ability. The ensemble's prediction for a new instance is determined by combining votes across all trees. For example, this can be done by using a majority vote for classification tasks and averaging for regression tasks; a description can be found in <d-cite key="murphy-2012"></d-cite>, <d-cite key="rokach-2016"></d-cite>, <d-cite key="weinberger-2018"></d-cite>.

It is also possible to employ early stopping or pruning for the individual trees. In particular, decision trees based on a single decision are called decision tree stumps. For the pruning step, instead of growing the tree to its full depth based on data $S_j$, we can use the left-out samples $S \setminus S_j$ as a validation set and post-prune the decision tree. <d-cite key="weinberger-2018"></d-cite>

An advantage of the random forest is that it only has two hyperparameters: the number of subsampled datasets $M$ and the number of subsampled features $k$; it is very insensitive to both <d-cite key="weinberger-2018"></d-cite>.


## Extremely Randomized Trees

Extremely randomized trees (Extra-Trees) introduce a high degree of randomization in selecting splits for both attributes and cut points. Unlike conventional methods that seek to find the optimal split based on specific criteria (such as information gain), Extra-Trees selects these splits totally or partially at random. The extreme case is that the structure of the built tree does not correlate at all with the labels of the training set. The benefit of this method is that it is computationally very efficient to implement. <d-cite key="geurts-2006"></d-cite>

Traditional tree models are significantly influenced by the randomness of the learning sample, leading to a high variance in the model outcomes. The Extra-Trees algorithm aims to leverage this randomness as a core component of the model-building process. By selecting splits randomly, the method seeks to reduce the variance associated with the choice of cut points, which has been identified as a significant contributor to the error rates in tree-based methods. In addition, this algorithm is computationally very efficient despite the necessity of growing multiple models. <d-cite key="rokach-2016"></d-cite> <d-cite key="geurts-2006"></d-cite>

The ensemble model for the Extra-Trees has two parameters: $k$, the number of features randomly selected for each node, and the early stopping criterion $n_\mathrm{min}$ that is, if the subset $S$ has strictly fewer samples than $n_\mathrm{min}$ the algorithm terminates. In addition, the algorithm does not use bootstrapped sub-samples as for the random forest but instead considers the whole input data set $M$ times. As usual, the final classification is determined using a majority vote. <d-cite key="geurts-2006"></d-cite>

The corresponding algorithm to build and ExtraTree is defined using that for a given set $S$, $\operatorname{values}_S(i) = \operatorname{values} \\{ \mathbf{x} \in \mathcal{S} \mid \mathbf{x}_i \\}$ is the set of all possible values of the i'th feature in $\mathcal{S}$.

**Extra-Tree Training Algorithm**, *adapted from <d-cite key="geurts-2006"></d-cite>*
$$
\begin{array}{l}
\textbf{Input:} &\text{Training set } S \subseteq \mathcal{D} = \{(\mathbf{x}_i, y_i) \mid \mathbf{x}_i \in \mathbb{R}^d, y_i \in \{-1, 1\}\}_{i=1}^n \\
\textbf{Output:} &\text{Decision tree}
\end{array}
$$
$$
\begin{array}{l}
\textbf{Procedure: } \text{Extra-Trees}(S) \\
\quad \textbf{if } |S| < n_\text{min} \textbf{ or all } y_i \text{ in } S \text{ are identical or all } \vec{x}_i \text{ in } S \text{ are identical} \\
\quad \quad \textbf{return } \operatorname{Leaf}(\operatorname{mode}(Y)), \text{ where } Y = \{y_1, \ldots, y_n\} \text{ and ties are broken randomly} \\
\quad \textbf{else} \\
\quad \quad \text{Sample a random subset of features } I \subseteq [d] \text{ of size } |I| = k \leq d \text{ without replacement} \\
\quad \quad \textbf{for each attribute } i \text{ in } I \textbf{ do} \\
\quad \quad \quad s_i = \text{Pick a Random Split}(S, i) \\
\quad \quad \textbf{end for} \\
\quad \quad \text{Find } s^\star \text{ such that } s^\star = \underset{i=1, \dots, k}{\arg \max} \operatorname{InfoGain}(S,s_i) \\
\quad \quad \text{Split } S \text{ into subsets } S_{\text{left}} \text{ and } S_{\text{right}} \text{ according to split } s^\star \\
\quad \quad T_{\text{left}} \leftarrow \text{Extra-Tree}(S_{\text{left}}) \\
\quad \quad T_{\text{right}} \leftarrow \text{Extra-Tree}(S_{\text{right}}) \\
\quad \quad \textbf{return } \text{tree with decision node on } s^\star, \text{ left branch } T_{\text{left}}, \text{ and right branch } T_{\text{right}} \\
\quad \textbf{end if}
\end{array}
$$

$$
\begin{array}{l}
\textbf{Input: } &\text{Whole Training Set } \mathcal{D}, \text{ Training Set } S \subseteq \mathcal{D}, \text{ Feature } i \\
\textbf{Output: } & \text{Split } s_i \text{ on feature } i \\
\end{array}
$$
$$
\begin{array}{l}
\textbf{Procedure: } \text{Pick a Random Split}(S, i) \\
\quad \text{Let } \mathcal{A}_\mathcal{D} = \operatorname{values}_{\mathcal{D}}(i) \text{ be the set of all possible values for } i \text{ in } \mathcal{D} \\
\quad \text{Let } \mathcal{A}_S = \operatorname{values}_S(i) \subseteq \mathcal{A}_\mathcal{D}, \text{ i.e., the subset of } \mathcal{A}_\mathcal{D} \text{ values that appears in } S \\
\quad \text{Randomly draw a proper non-empty subset } \mathcal{A}_1 \subset \mathcal{A}_S \text{ and a subset } \mathcal{A}_2 \subset \mathcal{A}_\mathcal{D} \setminus \mathcal{A}_S \\
\quad \textbf{return } \text{split } s_i \in \mathcal{A}_1 \cup \mathcal{A}_2 \text{ on feature } i
\end{array}
$$

## Gradient Boosting

The main idea behind boosting is to go from a weak classifier to a strong classifier. Given a current ensemble, one way to approach this problem is to train another classifier where misclassified samples are given a higher importance. This is known as Adaboost <d-cite key="adaboost"></d-cite>. Instead of giving misclassified samples higher importance, you can also sequentially train new classifiers to approximate the residuals of the previous ensemble. Adding them to the previous ensemble will "correct" it and is known as gradient boosting. This section presents gradient boosting based on the work in <d-cite key="gbm_orig"></d-cite>.

### Formal Definition

Our goal is to find a function $\hat{F} \in \mathcal{F}$ that best approximates $y$ given $x$. A function $F \in \mathcal{F}$ is of the form for some $M \in \mathbb{N}$: 

$$F(x) = \sum_{m=1}^M h_m(x) \gamma_m + c = \sum_{m=0}^M \gamma_m h_m(x)$$

For simplicity of notation, define $\gamma_0 h_0(x) = c$. We have that $\gamma_1, \dots, \gamma_n, c \in \mathbb{R}$ are constants and $h_1, ..., h_m \in \mathcal{H}$ are known as base learners. For example, these base learners could be decision trees or linear regression models.

We will define:

$$
\begin{align}
    &F_0(x) = c = \underset{a}{\arg \min } \mathbb{E}_{x, y} \left[\mathcal{L}(y, a)\right]
\end{align}
$$

We further define for $m \geq 1$ and some differentiable loss function $\mathcal{L}$:

$$
\begin{align}
    r_m &= - \nabla_{F_{m-1}} \mathcal{L}(y, F_{m-1}(x)) \label{math_res}\\
    h_m &= \underset{h \in \mathcal{H}}{\arg \min }  \mathbb{E}_{x, y}\left[(r_m - h(x))^2\right] \label{math_h} \\
    \gamma_m &= \underset{\gamma}{\arg \min } \mathbb{E}_{x, y}[\mathcal{L}\left(y, F_{m-1}(x) + \gamma h_m(x)\right)] \label{math_gamma} \\
    F_m(x) &= F_{m-1}(x) + \gamma_m h_m(x) \label{math_F}
\end{align}
$$

In the literature, $r_m$ is generally referred to as the pseudo-residual. For a given parameter $M \in \mathbb{N}$, we will define our approximation $\hat{F}(x) = F_M(x)$. 

### Connection to Steepest Gradient Descent

Consider the following optimization problem:

$$
\begin{align}
    &h_m=\underset{h \in \mathcal{H}}{\arg \min } \mathbb{E}_{x, y} \left[\mathcal{L}\left(y, F_{m-1}\left(x\right)+h\left(x\right)\right)\right] \label{math_opt_orig}
\end{align}
$$

For a general function class $\mathcal{H}$, the optimization problem in equation $$\ref{math_opt_orig}$$ is infeasible.

As we have seen in the lecture, we can find a local minimum of a function with steepest gradient descent <d-cite key="gd_steepest"></d-cite>. Given a function $F_{m-1} \in \mathcal{F}$, an iteration of steepest gradient descent with the aim of minimizing $\mathcal{L}$ yields:

$$
\begin{align}
    \hat{\gamma}_m = \underset{\gamma}{\arg \min } \mathbb{E}_{x, y}\left[\mathcal{L}\left(y, F_{m-1}(x) - \gamma \nabla_{F_{m-1}} \mathcal{L}(y, F_{m-1}(x)) \right) \right]\\
    \hat{F}_m(x) = F_{m-1}(x) - \gamma_m \mathbb{E}_{x, y}\left[\nabla_{F_{m-1}} \mathcal{L}(y, F_{m-1}(x))\right]
\end{align}
$$

By the properties of steepest gradient descent, we have 

$$\mathbb{E}_{x, y}\left[\mathcal{L}(y, \hat{F}_m(x))\right] \leq \mathbb{E}_{x, y}\left[\mathcal{L}(y, F_{m-1}(x))\right]$$

However, we do not necessarily have $$\hat{F}_m \in \mathcal{F}$$ given that $F_{m-1} \in \mathcal{F}$.

Equation $$\ref{math_h}$$ defines an approximation in $\mathcal{H}$ of $$- \mathbb{E}_{x, y}\left[\nabla_{F_{m-1}} \mathcal{L}(y, F_{m-1}(x)) \right]$$. Consequently, we can understand gradient boosting as an approximation of the steepest gradient descent method, where we approximate the gradient by some function $h \in \mathcal{H}$. 

### Algorithm for a finite Dataset
Consider a finite dataset $\mathcal{D} = \\{ (x_i, y_i) \mid  i = 1, \dots, n \\} $:

**Gradient Boosting**, *adapted from <d-cite key="gbm_orig"></d-cite>* \\
$$
\begin{array}{l}
F_0(x) = \underset{a}{\arg \min } \mathbb{E}_{y}[\mathcal{L}(y, a)] \\
\textbf{for } m = 1 \text{ to } M \textbf{ do} \\
\quad r_i = - \nabla_{F_{m-1}} \mathcal{L}(y_i, F_{m-1}(x_i)) \text{ for } i=1,\dots, n \\
\quad h_m = \underset{h \in \mathcal{H}}{\arg \min } \sum_{i=1}^n (r_i - h(x_i))^2 \\
\quad \gamma_m = \underset{\gamma}{\arg \min } \sum_{i=1}^n \mathcal{L}(y_i, F_{m-1}(x_i) + \gamma h(x_i)) \\
\quad F_m(x) = F_{m-1}(x) + \gamma_m h_m(x) \\
\textbf{end for} \\
\textbf{return } F_M(x)
\end{array}
$$

### Regularization

Compared to the random forest, boosting iteratively "corrects" ensemble models over the course of training. Although this might lead to higher performance on the dataset, this might hurt generalization since we might overfit the training data by "over-correcting" for it. This becomes especially problematic if our training data is noisy or even contains outliers; since then, we will be fitting the noise. Consequently, modern implementations <d-cite key="scikit-learn"></d-cite>, <d-cite key="lgbm"></d-cite>, <d-cite key="xgboost"></d-cite> have specific parameters for regularization.

#### Regularize Base Learners

Similar to other machine learning methods, we can add a regularization parameter $\Omega$ that penalizes the complexity of the model. For example, we could adapt the definition of $h_m$ as follows:

$$
h_m = \underset{h \in \mathcal{H}}{\arg \min } \mathbb{E}_{x, y}\left[(r - h(x))^2\right] + \Omega(h)
$$

#### Early Stopping

Early Stopping is a method that is used to estimate the number of weak learners required for training. Before training the model, we set aside a part of the training data, which we will refer to as the validation data. We evaluate our model on this validation set by adding new estimators during training. As soon as the performance on the validation set decreases or does not improve, we will stop training. This means that no new estimators are added.

#### Shrinkage/Learning Rate

We can add a parameter $\nu$, which scales the contribution of each weak learner. The update in equation $$\ref{math_opt_orig}$$ will be changed to:

$$
F_m(x) = F_{m-1}(x) + \nu \cdot \gamma_m h_m(x)
$$

In literature <d-cite key="gbm_orig"></d-cite>, this parameter is called shrinkage. However, many implementations, such as the one from `scikit-learn` <d-cite key="scikit-learn"></d-cite>, call this parameter `learning_rate` instead. Similar to gradient descent, where a lower learning rate will result in slower convergence, we will need more estimators to converge. Empirical results have shown that $\nu \leq 0.1$ combined with early stopping achieves the best results <d-cite key="scikit-learn"></d-cite>.

#### Subsampling

Stochastic Gradient Boosting <d-cite key="gbm_stochastic"></d-cite> is a minor modification to the algorithm. It consists of training each base learner only on a subsample of the training data. In `scikit-learn` <d-cite key="scikit-learn"></d-cite>, this parameter is called `subsample`. We have that $0 <$ `subsample` $\leq 1$.

### Gradient tree boosting

Consider the case where we use decision trees as base learners. $h_m$ is of the following form:

$$
h_m(x) = \sum_{j=1}^{J_m} b_{j,m} \mathbb{1}_{R_{j,m}}(x)
$$

The decision tree splits the input space into $J_m$ distinct regions $R_{1,m},\dots,R_{J_m,m}$. Furthermore, $b_{j,m}$ denotes the value predicted in the $R_{j,m}$ region. $$\mathbb{1}_{R_{j,m}}(x)$$ is the indicator value for $x \in R_{j,m}$.

#### Improved Optimization
Another version <d-cite key="gbm_orig"></d-cite> alters equation $$\ref{math_gamma}$$ and $$\ref{math_F}$$ to:

$$
\begin{align}
    & \gamma_{j,m} = \underset{\gamma}{\arg \min }  \mathbb{E}_{x \in R_{j,m}, y}[\mathcal{L}\left(y, F_{m-1}(x) + \gamma h_m(x)\right)] \\
    & F_m(x) = F_{m-1}(x) + \sum_{j=1}^{J_m} \gamma_{j,m} \mathbb{1}_{R_{j,m}}(x)
\end{align}
$$

In essence, we have a separate error correction for each region. This is guaranteed to not give a worse solution on a single iteration since the solution to the original optimization problem considers the same exact problem but with the constraint of $\gamma_{1,m} = \dots = \gamma_{J_m,m}$.

#### Histogram Gradient Boosting
By the iterative design of gradient boosting, the parallelization of the training process is much harder compared to the random forest. In the case of the random forest, we can naturally build multiple trees in parallel, whereas this is not possible in the case of boosting. This limited the use of the classical gradient boosting algorithm on large datasets since the training process took too long. Histogram Gradient Boosting can be used to increase the speed of building individual trees.

The main bottleneck in building a decision tree is finding the optimal split. This requires us to compute a metric for all features and feature values, which requires $\mathcal{O}(\text{\#features} \times n \log n)$ time since we need to sort all values of all our features.

Although we cannot increase the sorting speed, we can approximate the optimal split by quantizing each feature. This is done by splitting the data values of each feature into bins. These bins are equal-density histograms, meaning all the intervals contain the same number of values. The quantized value of the feature will be the index of the interval. This index is chosen so that the ordinal order is preserved between the values of features in different intervals. Here is an example for the case where we have three bins, four samples, and two features <d-cite key="histogram_gbm"></d-cite>:

$$
\begin{align*}
    \begin{bmatrix}
    1.5 & 0.0\\
    0.0 & 5.5\\
    0.3 & 7.0\\
    5.5 & 8.5
    \end{bmatrix}
    \quad\rightarrow\quad
    \begin{bmatrix}
    1 & 0\\
    0 & 1\\
    0 & 2\\
    2 & 2
    \end{bmatrix}
\end{align*}
$$

The best split is then computed by not considering the feature values but the histogram interval values. Since the number of intervals is a constant value and the indexes are chosen to be the values $$1, 2, \dots, \#\text{intervals}$$, we are not required to sort the values in order to compute the gain. This results in a runtime of $\mathcal{O}(\text{\#features} \times \text{\#intervals})$.

## XGBoost

XGBoost is an open-source implementation of regularized Gradient Boosting Machines <d-cite key="xgboost"></d-cite>. It has been and is still being used extensively in machine learning competitions, where it is commonly part of the winning solution. However, many people use this model as a black box without properly understanding the underlying principles. While discussing all of the implementation details of XGBoost would go beyond this blog post, this section aims to give a broad overview of the main ideas behind XGBoost <d-cite key="xgboost"></d-cite>.

### Second Order Gradient Boosting
XGBoost is based on the ideas behind gradient boosting. However, in contrast to the standard algorithm described in the previous section, XGBoost considers Gradients and Hessians.

### Formal Definition
Similar to gradient boosting, our goal is to find a function $\hat{F} \in \mathcal{F}$ that best approximates $y$ given $x$. A function $F \in \mathcal{F}$ is of the form for some $M \in \mathbb{N}$:

$$
F(x) = \sum_{m=1}^M h_m(x) \gamma_m + c = \sum_{m=0}^M \gamma_m h_m(x)
$$

For simplicity of notation, define $\gamma_0 h_0(x) = c$. We have that $$\gamma_1, \dots, \gamma_n, c \in \mathbb{R}$$ are constants and $h_1, \dots, h_m \in \mathcal{H}$ are known as base learners.

We will define:

$$
F_0(x) = c = \underset{a}{\arg \min } \mathbb{E}_{x, y} [\mathcal{L}(y, a)]
$$

We further define for $m \geq 1$ and some twice differentiable loss function $\mathcal{L}$:

$$
\begin{align}
    g &= \nabla_{F_{m-1}(x)} \mathcal{L}(y, F_{m-1}(x)) \label{math_res_2}\\
    H &= \nabla^2_{F_{m-1}(x)} \mathcal{L}(y, F_{m-1}(x)) \label{math_res_2H}\\
    h_m &= \underset{h \in \mathcal{H}}{\arg \min } \mathbb{E}_{x, y}\left[\frac{H}{2} \left(\frac{g}{H} - h(x)\right)^2\right] \label{math_h_2}\\
    F_m(x) &= F_{m-1}(x) + h_m(x) \label{math_F_2}
\end{align}
$$

For a given parameter $M \in \mathbb{N}$, we will define our approximation $\hat{F}(x) = F_M(x)$.

### A second order approximation of $\mathcal{L}$

Consider the following optimization problem:

$$
\begin{align}
    &h_m= \underset{h \in \mathcal{H}}{\arg \min } \mathbb{E}_{x, y} \left[\mathcal{L}\left(y, F_{m-1}\left(x\right)+h\left(x\right)\right)\right] \label{math_opt_orig_2}
\end{align}
$$

For a general function class $\mathcal{H}$, the optimization problem in equation $$\ref{math_opt_orig_2}$$ is infeasible. We will use the second-order Taylor approximation of the loss function.

$$
\begin{align}
    & \mathcal{L}\left(y, F_{m-1}\left(x\right)+h\left(x\right)\right) \approx \mathcal{L}(y, F_{m-1}(x)) + g h(x) + \frac{1}{2} H h(x)^2
\end{align}
$$

The optimization problem for this approximation can be rewritten as follows:

$$
\begin{align}
    h_m
    &= \underset{h \in \mathcal{H}}{\arg \min } \mathbb{E}_{x, y} \left[\mathcal{L}(y, F_{m-1}(x)) + g h(x) + \frac{1}{2} H h(x)^2\right] \\
    &= \underset{h \in \mathcal{H}}{\arg \min } \mathbb{E}_{x, y} \left[g h(x) + \frac{1}{2} H h(x)^2 + \frac{1}{2}g^2\right] \\
    &= \underset{h \in \mathcal{H}}{\arg \min } \mathbb{E}_{x, y} \left[\frac{H}{2} \left(\frac{g}{H} - h(x)\right)^2 \right]
\end{align}
$$

This is exactly the same optimization problem as in equation $$\ref{math_h_2}$$. It is worth mentioning that XGBoost adds regularization. This is done by adding $\Omega(h)$ to the minimization problem for $h_m$, which penalizes complexity. Furthermore, [Shrinkage/Learning Rate](#shrinkage) from Gradient Boosting and feature subsampling from the Random Forest are used.

### Algorithm for a finite Dataset

**Second Order Gradient Boosting**, *Adapted from <d-cite key="xgboost"></d-cite>*

Consider a finite dataset $\mathcal{D} = \\{(x_i, y_i) \mid i = 1, \dots, n\\}$:
$$
\begin{array}{l}
F_0(x) = \underset{a}{\arg \min}  \sum_{i=1}^n \mathcal{L}(y_i, a) \\
\textbf{for } m = 1 \text{ to } M \textbf{ do} \\
\quad g_i = \nabla_{F_{m-1}} \mathcal{L}(y, F_{m-1}(x)) \text{ for } i = 1, \dots, n \\
\quad H_i = \nabla^2_{F_{m-1}} \mathcal{L}(y, F_{m-1}(x)) \text{ for } i = 1, \dots, n \\
\quad h_m = \underset{h \in \mathcal{H}}{\arg \min } \sum_{i=1}^n \frac{H_i}{2}\left(\frac{g_i}{H_i} - h(x_i) \right)^2 \\
\quad F_m(x) = F_{m-1}(x) + h_m(x) \\
\textbf{end for} \\
\textbf{return } F_M(x)
\end{array}
$$

### Weighted Quantile Sketch

For tree building, XGBoost uses an approximate solution that shares some similarities with [Histogram Gradient Boosting](#histogram-gradient-boosting). The main goal is to reduce the time needed to find the optimal split. Consider the multi-set $$\mathcal{D}_k = \{(x_{1,k}, H_1), (x_{2,k}, H_2), \ldots, (x_{n,k}, H_n)\}$$, where $x_{i,k}$ is the $i$'th feature value of the $k$'th feature and $H_i$ is equation $$\ref{math_F_2}$$ evaluated at point $x_i$. We will define the following rank function:

$$
r_k(z) = \frac{1}{\sum_{(x,H) \in \mathcal{D}_k} H} \sum_{(x, H) \in \mathcal{D}_k, x < z} H
$$

Intuitively, this ranking function can be interpreted as the data points weighted by $H_i$. This happens in equation $$\ref{math_h_2}$$ as well, which is nothing else but a weighted sum of the squared loss with labels $\frac{g_i}{H_i}$ and weights $H_i$.

A candidate split for $\epsilon > 0$ of the $k$'th feature is defined as the data points $$\{s_{k,1}, \dots, s_{k,l}\}$$, such that

$$
|r_k(s_{k,j}) - r_k(s_{k,j+1})| < \epsilon, \quad s_{k,1} = \min_i x_{i,k}, \quad s_{k,l} = \max_i x_{i, k}
$$

Finding such a candidate split is called the weighted quantile sketch problem, which is non-trivial for large datasets. The authors of <d-cite key="xgboost"></d-cite> propose an algorithm to approximate the problem. This algorithm has some nice theoretical properties. The proofs and description of the algorithm can be found in the appendix of <d-cite key="xgboost"></d-cite>.

### Sparsity Awareness

Many real-world datasets are sparse, which is generally caused by missing values in the data, frequent zero values in statistics, and artifacts from feature engineering (e.g., one-hot encoding).

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/ensemble-methods/xgboost_sparsity.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

*Figure: Handling Missing Values. Reproduced from <d-cite key="xgboost"></d-cite>*

XGBoost adds a default direction for missing values in the decision tree. In order to generate a new split for a feature, only the non-missing values are considered. However, the split is computed for both cases where the missing values either all go to the left or the right direction. The optimal split and the default direction for the missing values are chosen based on the maximum gain. For certain very sparse datasets, this can lead to a 50x improvement in runtime <d-cite key="xgboost"></d-cite> compared to a basic solution such as imputation.

### System Design

Here are some features of XGBoost with regard to System Design. For more details, see <d-cite key="xgboost"></d-cite>.

- Parallelization
- Cache-aware Computation, which results in a 2x improvement in runtime for very large datasets <d-cite key="xgboost"></d-cite>
- Distributed Training on a Cluster
- Support for processing large datasets that do not fit onto the main disk

## Experiment

In this section, we present the results of Random Forest, Extra Trees, Gradient Boosting, and XGBoost on real-world data. We use the implementations of `scikit-learn` <d-cite key="scikit-learn"></d-cite> and `xgboost` <d-cite key="xgboost"></d-cite> for our experiments; the corresponding implementation can be found in [Code for Experiment](#code-for-experiment). This section demonstrates the practical ease of use of these algorithms. This often leads to them being used as a black box.

### California Housing
We use the `California Housing` dataset from `scikit-learn` <d-cite key="scikit-learn"></d-cite>. This dataset consists of only eight numeric features and is a regression task consisting of predicting a house price given these features. We perform 3-fold cross-validation on the dataset and measure runtime & `R2 Score`. Furthermore, we train multiple instances of each classifier, where we modify the maximum number of estimators each method uses. The error bars denote the difference in runtime between different folds.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/ensemble-methods/xgboost_experiment.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

*Figure: Runtime and R2 Score with different number of trees. The left figure presents an overview of all evaluated models, whereas the right excludes the Gradient Boosting model to enhance readability.*

As one can clearly see, the runtime of standard gradient boosting is significantly longer compared to the other methods. This can be explained by the fact that the iterative tree building of gradient boosting is hard to parallelize. Since the experiments were run on a machine with 128 cores, this significantly impacts runtime. Another observation that can be made is that the Random Forest and Extra Trees both plateau much earlier than the gradient methods while having the shortest runtime. 

## Appendix

### Code for Experiment

```python
# This code has been adapted from: 
# https://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_hist_grad_boosting_comparison.html

from sklearn.datasets import fetch_california_housing

X, y = fetch_california_housing(return_X_y=True, as_frame=True)
n_samples, n_features = X.shape

print(f"The dataset consists of {n_samples} samples and {n_features} features")

import joblib

N_CORES = 128#joblib.cpu_count(only_physical_cores=True)
print(f"Number of physical cores: {N_CORES}")

import pandas as pd

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV, KFold

models = {
    "Extra Trees": ExtraTreesRegressor(
        random_state=0, n_jobs=N_CORES, #max_depth=8, min_samples_leaf=5, min_samples_split=3, min_weight_fraction_leaf=0.1, 
    ),
    #"Decision Tree" : DecisionTreeRegressor(
    #   random_state=0
    #),
    "XGBoost" : XGBRegressor(
        random_state=0, learning_rate=0.1, max_depth=5,
    ),
    "Gradient Boosting" : GradientBoostingRegressor(
        random_state=0, max_depth=8, learning_rate=0.2, min_samples_split=5, min_samples_leaf=3, min_weight_fraction_leaf=0.1
    ),
    "Random Forest": RandomForestRegressor(
        random_state=0, n_jobs=N_CORES, #max_depth=10, min_samples_leaf=2, min_samples_split=5, min_weight_fraction_leaf=0.1
    ),

    #"Hist Gradient Boosting": HistGradientBoostingRegressor(
    #    max_leaf_nodes=15, random_state=0, early_stopping=False
    #),
}
n_estimators = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
max_depth = [5, 6, 7, 8, 9, 10]
learning_rate = [0.01, 0.05, 0.1, 0.15, 0.2]
min_samples_split = [2, 3, 4, 5, 6]
min_samples_leaf = [1,2,3,4,5]
min_weight_fraction_leaf = [0.1, 0.2, 0.3, 0.4, 0.5]

param_grids = {
    "Extra Trees" : {
        "n_estimators": n_estimators,
        #'max_depth' : max_depth, 
        #'min_samples_split' : min_samples_split, 
        #'min_samples_leaf' : min_samples_leaf, 
        #'min_weight_fraction_leaf' : min_weight_fraction_leaf
    },
    "XGBoost": {
        "n_estimators": n_estimators,
        #'max_depth' : max_depth, 
        #'learning_rate' : learning_rate, 
        #'min_samples_split' : min_samples_split, 
        #'min_samples_leaf' : min_samples_leaf, 
        #'min_weight_fraction_leaf' : min_weight_fraction_leaf
    },
    "Gradient Boosting": {
        "n_estimators": n_estimators, 
        #'max_depth' : max_depth, 
        #'learning_rate' : learning_rate, 
        #'min_samples_split' : min_samples_split, 
        #'min_samples_leaf' : min_samples_leaf, 
        #'min_weight_fraction_leaf' : min_weight_fraction_leaf
    },
    "Random Forest": {
        "n_estimators": n_estimators,
        #'max_depth' : max_depth, 
        #'min_samples_split' : min_samples_split, 
        #'min_samples_leaf' : min_samples_leaf, 
        #'min_weight_fraction_leaf' : min_weight_fraction_leaf
    },
    "Hist Gradient Boosting": {
        "max_iter": n_estimators
    },
}

cv = KFold(n_splits=5, shuffle=True, random_state=42)

results = []
for name, model in models.items():
    print(name)
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grids[name],
        return_train_score=True,
        cv=cv,
        n_jobs=1 if name != "Gradient Boosting" else -1, # train multiple models if gradient boosting since it will use at most 1 core anyway. For the other methods, only train 1 model at a time in order to use all cores for training a single model
    ).fit(X, y)
    result = {"model": name, "cv_results": pd.DataFrame(grid_search.cv_results_)}
    results.append(result)

    import plotly.colors as colors
import plotly.express as px
from plotly.subplots import make_subplots

fig = make_subplots(
    rows=1,
    cols=1,
    subplot_titles=["Train time vs score"],
    vertical_spacing=0.1,
    row_heights=[5.0]
)

model_names = [result["model"] for result in results]
colors_list = colors.qualitative.Plotly * (
    len(model_names) // len(colors.qualitative.Plotly) + 1
)

for idx, result in enumerate(results):
    cv_results = result["cv_results"].round(3)
    model_name = result["model"]
    param_name = list(param_grids[model_name].keys())[0]
    cv_results[param_name] = cv_results["param_" + param_name]
    cv_results["model"] = model_name

    scatter_fig = px.scatter(
        cv_results,
        x="mean_fit_time",
        y="mean_test_score",
        error_x="std_fit_time",
        error_y="std_test_score",
        hover_data=param_name,
        color="model",
    )
    line_fig = px.line(
        cv_results,
        x="mean_fit_time",
        y="mean_test_score",
    )

    scatter_trace = scatter_fig["data"][0]
    line_trace = line_fig["data"][0]
    scatter_trace.update(marker=dict(color=colors_list[idx]))
    line_trace.update(line=dict(color=colors_list[idx]))
    fig.add_trace(scatter_trace, row=1, col=1)
    fig.add_trace(line_trace, row=1, col=1)

fig.update_layout(
    xaxis=dict(title="Train time (s) - lower is better"),
    yaxis=dict(title="Test R2 score - higher is better"),
    legend=dict(x=0.72, y=0.05, traceorder="normal", borderwidth=1),
    title=dict(x=0.5, text="Speed-score trade-off"),
    height=1000,
    width=800
)

fig.write_image("train_time_vs_score.svg")
fig.write_image("train_time_vs_score.png")
fig.show()


import plotly.colors as colors
import plotly.express as px
from plotly.subplots import make_subplots

fig = make_subplots(
    rows=1,
    cols=1,
    subplot_titles=["Train time vs score"],
    vertical_spacing=0.1,
    row_heights=[5.0]
)

model_names = [result["model"] for result in results]
colors_list = colors.qualitative.Plotly * (
    len(model_names) // len(colors.qualitative.Plotly) + 1
)

for idx, result in enumerate(results):

    cv_results = result["cv_results"].round(3)
    model_name = result["model"]
    if model_name == 'Gradient Boosting': continue
        
    param_name = list(param_grids[model_name].keys())[0]
    cv_results[param_name] = cv_results["param_" + param_name]
    cv_results["model"] = model_name

    scatter_fig = px.scatter(
        cv_results,
        x="mean_fit_time",
        y="mean_test_score",
        error_x="std_fit_time",
        error_y="std_test_score",
        hover_data=param_name,
        color="model",
    )
    line_fig = px.line(
        cv_results,
        x="mean_fit_time",
        y="mean_test_score",
    )

    scatter_trace = scatter_fig["data"][0]
    line_trace = line_fig["data"][0]
    scatter_trace.update(marker=dict(color=colors_list[idx]))
    line_trace.update(line=dict(color=colors_list[idx]))
    fig.add_trace(scatter_trace, row=1, col=1)
    fig.add_trace(line_trace, row=1, col=1)

fig.update_layout(
    xaxis=dict(title="Train time (s) - lower is better"),
    yaxis=dict(title="Test R2 score - higher is better"),
    legend=dict(x=0.72, y=0.05, traceorder="normal", borderwidth=1),
    title=dict(x=0.5, text="Speed-score trade-off"),
    height=1000,
    width=800
)

fig.write_image("train_time_vs_score_no_gb.svg")
fig.write_image("train_time_vs_score_no_gb.png")
fig.show()
```