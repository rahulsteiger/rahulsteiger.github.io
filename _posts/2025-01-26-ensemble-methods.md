---
layout: distill
title: Ensemble Methods
description: Ensemble methods combine multiple simple learning algorithms to achieve superior overall performance. This blog post is an adaptation of a group project I completed during one of my courses at the National University of Singapore. 
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
  - name: Experiments
  - name: Code

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

Ensemble methods combine multiple simple learning algorithms to produce an overall better result. First, we will provide a brief overview of weak and strong learners and the concept of a decision tree. In the next step, we will introduce two ensemble methods based on bagging: random forests and extremely randomized trees. The second part of the paper will detail gradient boosting algorithms, focusing on introducing the conventional Gradient Boosting Machine and discussing the principle ideas behind the notoriously popular method XGBoost. Finally, we will conclude with a comparative study of the performance of these models on a representative sample dataset. 


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

Consequently, we can reduce the variance arbitrarily by just using more estimators. However, assuming that $f_1, \dots, f_n$ are mutually independent is unrealistic if we assume they are trained on the same dataset. In this blog, we will discuss algorithms that combine multiple weak learners while ensuring that $\operatorname{Cov}[\hat{f}_i(x), \hat{f}_j(x)] \approx 0$ and a sufficiently low bias.

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
\quad \text{Select feature } j \text{ maximizing information gain: } j = \arg\max_{i \in F} \operatorname{InfoGain}(S, i) \\
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
\quad \quad \text{Find } s^\star \text{ such that } s^\star = \arg\max_{i=1, \dots, k} \operatorname{InfoGain}(S,s_i) \\
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