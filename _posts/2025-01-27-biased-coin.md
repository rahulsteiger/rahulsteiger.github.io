---
layout: distill
title: Biased Coin
description: An analytical solution to a coin-tossing problem with an unknown bias.

tags: coin-tosses probability quantitative-trading
giscus_comments: false
date: 2025-01-27
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
  - name: Rahul Steiger
    affiliations:
      name: ETH Zurich

bibliography:

# Optionally, you can add a table of contents to your post.
# NOTES:
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
#   - we may want to automate TOC generation in the future using
#     jekyll-toc plugin (https://github.com/toshimaru/jekyll-toc).
toc:
  - name: Problem Statement
    # if a section has subsections, you can add them as follows:
    # subsections:
    #   - name: Example Child Subsection 1
    #   - name: Example Child Subsection 2
  - name: Strategy
  - name: A Useful Property
  - name: Expected Payoff
  - name: A Note on Attribution

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

## Problem Statement

A coin's probability of heads, $p$, is drawn once from the uniform distribution on $[0,1]$ and remains fixed. You do not know $p$. Conditional on $p$, the coin tosses are independent.

Before each of $N$ tosses, you predict heads or tails. A correct prediction earns 1 dollar; an incorrect prediction loses 1 dollar. You observe the outcome after every toss. What strategy maximizes your expected total payoff, and what is that expectation?

## Strategy

Predict whichever outcome has occurred more often so far. If heads and tails are tied, either prediction is equally good; we will choose randomly with equal probability. This includes the first toss, when neither outcome has been observed.

To see why this is optimal, let $H_n$ be the number of heads in the first $n$ tosses. Under the uniform prior, the probability that the next toss is heads, given $H_n=h$, is

$$
q_n(h)
= \frac{\int_0^1 p\,p^h(1-p)^{n-h}\,dp}
       {\int_0^1 p^h(1-p)^{n-h}\,dp}
= \frac{h+1}{n+2}.
$$

This is the posterior predictive probability. It differs from the observed fraction $h/n$, but both favor heads exactly when $h>n/2$.

Predicting heads gives an expected payoff of $2q_n(h)-1$, while predicting tails gives $1-2q_n(h)$. We therefore choose heads when $h>n/2$ and tails when $h<n/2$. At a tie, $q_n(h)=1/2$, so the expected payoff is zero.

Our predictions do not affect future tosses or which outcomes we observe. Maximizing the expected payoff on each toss therefore also maximizes the expected total payoff.

## A Useful Property

Although the tosses are independent conditional on $p$, they are not independent after averaging over the unknown bias. In fact, the number of heads after $n$ tosses is uniform on $\{0,\ldots,n\}$:

$$
\Pr(H_n=h)
= \int_0^1 {n\choose h}p^h(1-p)^{n-h}\,dp
= \frac{1}{n+1}.
$$

There is a useful way to see this without evaluating the integral. Draw $U_0,\ldots,U_n$ independently and uniformly from $[0,1]$. Set $p=U_0$, and call toss $i$ heads if $U_i<U_0$. Conditional on $U_0$, this produces independent tosses with heads probability $p$.

The event $H_n=h$ means that exactly $h$ of the other values lie below $U_0$, so $U_0$ ranks $(h+1)$st among the $n+1$ values. Every rank is equally likely, giving probability $1/(n+1)$.

## Expected Payoff

Let $R_{n+1}$ denote the payoff on toss $n+1$, after observing $n$ tosses. Under the optimal strategy,

$$
\mathbb{E}[R_{n+1}\mid H_n=h]
= |2q_n(h)-1|
= \frac{|2h-n|}{n+2}.
$$

This expression handles ties directly: when $2h=n$, the expected payoff is zero. Averaging over the uniform distribution of $H_n$ gives

$$
a_n := \mathbb{E}[R_{n+1}]
= \frac{1}{(n+1)(n+2)}\sum_{h=0}^{n}|2h-n|.
$$

For even $n=2m$, the sum is

$$
\sum_{h=0}^{2m}|2h-2m|
= 4\sum_{j=1}^{m}j
= 2m(m+1).
$$

For odd $n=2m+1$, it is

$$
\sum_{h=0}^{2m+1}|2h-(2m+1)|
= 2\sum_{j=0}^{m}(2j+1)
= 2(m+1)^2.
$$

Consequently,

$$
a_n =
\begin{cases}
\dfrac{n}{2(n+1)}, & n\text{ even},\\[6pt]
\dfrac{n+1}{2(n+2)}, & n\text{ odd}.
\end{cases}
$$

In particular, $a_0=0$: the first prediction has no expected advantage. Subsequent expected payoffs come in pairs,

$$
a_{2k-1}=a_{2k}=\frac{k}{2k+1}
=\frac{1}{2}\left(1-\frac{1}{2k+1}\right),
\qquad k\geq 1.
$$

Let $S_N=\sum_{t=1}^{N}R_t$ be the total payoff over $N\geq1$ tosses. By linearity of expectation,

$$
\begin{aligned}
\mathbb{E}[S_N]
&= \sum_{n=0}^{N-1}a_n\\
&= \frac{1}{2}\sum_{k=1}^{\lfloor N/2\rfloor}
   \left(1-\frac{1}{2k+1}\right)
 + \frac{1}{2}\sum_{k=1}^{\lfloor(N-1)/2\rfloor}
   \left(1-\frac{1}{2k+1}\right)\\
&= \frac{N-1}{2}
 - \frac{1}{2}\sum_{k=1}^{\lfloor N/2\rfloor}\frac{1}{2k+1}
 - \frac{1}{2}\sum_{k=1}^{\lfloor(N-1)/2\rfloor}\frac{1}{2k+1}.
\end{aligned}
$$

Empty sums are zero. As simple checks, one toss has expected payoff $0$, two tosses have expected total payoff $1/3$, and three tosses have expected total payoff $2/3$.

For $N=100$, the expected total payoff is approximately **47.56 dollars**. This averages over both the randomly chosen bias and the toss outcomes; it is not a guaranteed profit for any particular coin.

As we observe more tosses, the expected payoff per toss approaches $1/2$ dollar. This matches the average payoff if we knew the bias in advance and always chose the more likely outcome:

$$
\int_0^1 |2p-1|\,dp=\frac{1}{2}.
$$

The strategy itself is simple. The extra work is in calculating how much we expect to earn while learning the unknown bias.

## A Note on Attribution

I originally encountered this problem in a group chat. After sharing my solution, someone submitted it in an interview process without crediting me. That experience made me more thoughtful about attribution when sharing work.
