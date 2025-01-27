---
layout: distill
title: Biased Coin
description: This blog post presents a nice analytical solution I came up with for a quantitative trading interview question and a valuable life lesson for me. 

tags: coin-tosses probability quantitative-trading
giscus_comments: true
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
  - name: Expected Payoff
  - name: A Valuable Life Lesson

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
Suppose $p$ is uniform between $[0, 1]$ and you throw a coin $N$ times with bias $p$. You have to guess at each toss what the face will be. If your guess correctly, you get $1$ dollar, otherwise $-1$ dollar. What is the optimal strategy and the expected payoff in that case?

## Strategy
Since $p$ is uniform on $[0, 1]$, it does not matter what we guess for the first flip. After a certain number of throws, our best estimate for $p$ is the number of heads divided by the current number of throws. Since we want to maximize the number of correct guess, we predict heads if we have seen more heads and tails if we saw more tails in the previous throws. If we saw an equal number of heads and tails, we just predict randomly. 

## A useful property

Suppose $X_0, ..., X_n$ are i.i.d uniformly distributed variables on $[0, 1]$. We define

$$Y_i = \begin{cases} 1 & \text{if } X_i < X_0 \\ 0 & \text{if } X_i > X_0 \end{cases}$$

Given $X_0$, we have that $Y_1, \ldots, Y_n$ are conditionally independent of each other. We get that

$$P[Y_1 + \ldots + Y_n = k | X_0] = {n \choose k} X_0^k (1 - X_0)^{n - k}$$

Consequently,

$$E[Y_1 + \ldots + Y_n = k] = \int_0^1 {n \choose k} x^k (1 - x)^{n - k} \, dx = \frac{1}{n + 1}$$

**Note:** The event $Y_1 + \ldots + Y_n = k$ is equivalent to saying that if $X_0, \ldots, X_n$ are ordered, then $X_0$ has position $k+1$. This happens with probability $\frac{1}{n + 1}$.

## Expected Payoff
Let $X_n$ denote the payoff after the $n$'th toss, and $H$ denote the number of heads we saw until now. If we have $\frac{H}{n} = 0.5$, we will randomly guess, meaning that our overall expected payoff will be $0$. We have that:

$$\mathbb{E}[X_n | p] = \mathbb{E}\left[X_n \bigg | \frac{H}{n} < 0.5, p\right] P\left[\frac{H}{n} < 0.5 | p\right] + \mathbb{E}\left[X_n \bigg | \frac{H}{n} > 0.5, p\right] P\left[\frac{H}{n} > 0.5 | p \right]$$

Since we predict heads if $\frac{H}{n} > 0.5$, we have that

$$\mathbb{E}\left[X_n \bigg | \frac{H}{n} < 0.5, p\right] = (1 - p) \cdot 1 + p \cdot (-1) = 1 - 2p$$

Since we have that $$P\left[\frac{H}{n} < 0.5 \left . \right \vert  p\right] = \sum_{k=0}^{\lfloor \frac{n}{2} \rfloor} {n \choose k} p^k (1-p)^{n - k}$$, we get that

$$ 
\begin{align*}
   \int_0^{1} 2p \cdot P\left[\frac{H}{n} < 0.5 | p \right] \cdot 1 dp
    &= 2 \int_0^{1} \sum_{k=0}^{\lfloor \frac{n}{2} \rfloor} {n \choose k} p \cdot (1-p)^k p^{n - k} dp \\
    &= 2 \sum_{k=0}^{\lfloor \frac{n}{2} \rfloor} {n \choose k} \int_0^{1} (1-p)^{k} p^{(n + 1 - k)} dp \\
    &= 2 \sum_{k=0}^{\lfloor \frac{n}{2} \rfloor} {n \choose k} \frac{1}{(n + 2) \cdot {n+1 \choose k + 1}} \\
    &= 2 \sum_{k=0}^{\lfloor \frac{n}{2} \rfloor} \frac{k+1}{(n+1)(n+2)} \\
    &= \frac{(\lfloor \frac{n}{2} \rfloor + 1) (\lfloor \frac{n}{2} \rfloor + 2)}{(n + 1) (n + 2)}
\end{align*}
$$

Furthermore, 

$$
\begin{align*}
    \int_0^{1} 1 \cdot P\left[\frac{H}{n} < 0.5 | p \right] 1 \cdot dp
    &= \int_0^{1} \sum_{k=0}^{\lfloor \frac{n}{2} \rfloor} \cdot {n \choose k} p^k (1 - p)^{n - k} dp \\
    &= \sum_{k=0}^{\lfloor \frac{n}{2} \rfloor} {n \choose k} \int_0^{1} p^{k} (1 - p)^{(n - k)} dp \\
    &= \sum_{k=0}^{\lfloor \frac{n}{2} \rfloor} \frac{1}{(n+1)} \\
    &= \frac{\lfloor \frac{n}{2} \rfloor + 1}{(n + 1)}
\end{align*}
$$

Consequently, 

$$
\begin{align*}
    \int_0^1 E\left[Y \bigg | \frac{H}{n} < 0.5, p\right] P\left[\frac{H}{n} < 0.5 | p\right] \cdot 1 dp 
    &= \frac{\lfloor \frac{n}{2} \rfloor + 1}{(n + 1)} - \frac{(\lfloor \frac{n}{2} \rfloor + 1) (\lfloor \frac{n}{2} \rfloor + 2)}{(n + 1) (n + 2)} \\
    &= \frac{(\lfloor \frac{n}{2} \rfloor+1) (n - \lfloor \frac{n}{2} \rfloor)}{(n+1)(n+2)}\\
    &= \frac{(\lfloor \frac{n}{2} \rfloor+1) (\lceil \frac{n}{2} \rceil)}{(n+1)(n+2)}
\end{align*}
$$

Since the case $\frac{H}{n} > 0.5$ is symmetric, we get that

$$
E[X_n] = 2 \frac{(\lfloor \frac{n}{2} \rfloor+1) (\lceil \frac{n}{2} \rceil)}{(n+1)(n+2)}
$$

Finally, we have that

$$
\begin{align*}
    E[X] 
    &= \sum_{n=0}^{N-1} E[X_n] \\
    &= \sum_{n=1}^{N-1} 2 \frac{(\lfloor \frac{n}{2} \rfloor+1) (\lceil \frac{n}{2} \rceil)}{(n+1)(n+2)} \\
    &= 2 \sum_{n=1, n \text{ odd}}^{N-1} \frac{(n + 1)^2/2^2}{(n+1)(n+2)} + 2 \sum_{n=1, n \text{ even}}^{N-1} \frac{\frac{n + 2}{2} \frac{n}{2}}{(n+1)(n+2)} \\
    &= \sum_{n=1, n \text{ odd}}^{N-1} \frac{n + 1}{2 (n+2)} + \sum_{n=1, n \text{ even}}^{N-1} \frac{n}{2(n+1)} \\
    &= \sum_{n=2, n \text{ even}}^{N} \frac{n}{2 (n+1)} + \sum_{n=1, n \text{ even}}^{N-1} \frac{n}{2(n+1)} \\
    &= \frac{1}{2} \sum_{k=1}^{\lfloor \frac{N}{2} \rfloor} \left[1 - \frac{1}{2k + 1} \right] + \frac{1}{2} \sum_{k=1}^{\lfloor \frac{N-1}{2} \rfloor} \left[1 - \frac{1}{2k + 1} \right]\\
    &= \frac{N-1}{2} - \frac{1}{2} \sum_{k=1}^{\lfloor \frac{N}{2} \rfloor} \frac{1}{2k + 1} - \frac{1}{2}  \sum_{k=1}^{\lfloor \frac{N-1}{2} \rfloor} \frac{1}{2k + 1}
\end{align*}
$$

For $N = 100$, we get that $E[X] \approx 47.55727465647559$. 

**Note:** I do not believe that it would have been feasible to come up with the closed-form solution during an actual interview. 

### A Valuable Life Lesson
For the record, I did not receive this question during an interview; rather, I obtained it from a group chat. After coming up with what I believe is an elegant solution, I shared it with a couple of acquaintances. One of those people actually encountered this question in their interview but did not inform the interviewer. While they were unable to come up with this solution during the interview, that person sent an email to the firm submitting my solution as their own work without my permission or giving me credit. This action most likely allowed them to skip an interview stage and propelled them to the final round. It took a lot of willpower not to report that person for plagiarism. However, let me just say that I was not particularly surprised or disappointed when that person did not receive an offer. 

Happy Ending? ¯\\\_(ツ)_/¯