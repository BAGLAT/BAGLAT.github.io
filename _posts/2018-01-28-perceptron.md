---
title: "Credit Risk Modeling"
date: 2020-06-30
tags: [finance, data science, credit risk, loan default scorecards]
header:
  image: "/images/perceptron/percept.jpg"
excerpt: "Finance, Data Science, Credit Risk, Loan Default Scorecards, Risk Management"
mathjax: "false"
---

### [GitHub Project](https://github.com/BAGLAT/Credit-Risk-Modeling)

Modeling Expected Loss to identify customers that can default on a given loan in future and estimated capital requirement (Capital Adequacy or regulatory capital)

[Data Source](https://www.kaggle.com/wendykan/lending-club-loan-data)

Expected Loss = PD * EAD * LGD where PD = Probability of default (Works on Logisitic Regression) LGD = Loss given default (Works on Linear Regression) EAD = Exposure at default (Works on Linear Regression)

Following techniques are used to build the statistical model:

* Weight of evidence
* Information value
* Fine classing
* Coarse classing
* Linear regression
* Logistic regression
* Area Under the Curve
* Receiver Operating Characteristic Curve
* Gini Coefficient
* Kolmogorov-Smirnov
* Assessing Population Stability

Here's an image:
<img src="{{ site.url }}{{ site.baseurl }}/images/perceptron/linsep.jpg" alt="linearly separable data">

Here's another image using Kramdown:
![alt]({{ site.url }}{{ site.baseurl }}/images/perceptron/linsep.jpg)

