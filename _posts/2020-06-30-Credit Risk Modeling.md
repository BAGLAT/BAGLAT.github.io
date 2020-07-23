---
title: "Credit Risk Modeling"
date: 2020-06-30
tags: [finance, data science, credit risk, loan default scorecards]
header:
  image: "/images/CRM/ModelsCovered.png"
excerpt: "Finance, Data Science, Credit Risk, Loan Default Scorecards, Risk Management"
mathjax: "true"
---

### [GitHub Project](https://github.com/BAGLAT/Credit-Risk-Modeling)

<span style="font-family:Papyrus; font-size:2em;"> Credit Risk is the default risk that a person who borrowed money either from an individual or in the form of government bond is not going to repay the loan. In other words, Credit Risk is the likelihood that a borrower would not repay their loan to the lender.</span>

There are three types of losses a financial institute can incur:
1.	Expected Loss (EL) is the amount a bank can expect to lose, on average, over a predetermined period when extending credits to its customers.
2.	Unexpected Loss (UL) occurs in adverse economic circumstances. It is the volatility of credit losses around its expected loss. Once a bank determines its expected loss, it sets aside credit reserves in preparation for unexpected losses.
3.	Exceptional Stress Loss – Highly unlikely losses result due to the severe economic downturn such as at the time for financial crisis of 2008.
In this project I am focusing on modelling Expected Loss.

Three components are there in Expected Loss:
1.	Probability of Default (PD) – Likelihood that someone will default on a loan.
2.	Exposure at default (EAD) - Amount outstanding to be paid.
3.	Loss given default (LGD) - Exposure at Default / Recovery from Loss if you sell the bond.
Expected loss = PD * EAD * LGD

The process of modelling expected loss is divided into modelling each of the PD, LGD and EAD components separately. The three main steps for modelling each component are : 
1.	Data preparation or pre-processing.
2.	Model creation and estimation.
3.	Out-of-sample model validation.

[Data Source](https://www.kaggle.com/wendykan/lending-club-loan-data)
Consists of complete loan data for all loans issued through the 2007-2015, including the current loan status (Current, Late, Fully Paid, etc.) and latest payment information in USA.

Let’s begin our journey, below are the links to the notebooks explaining each model and building default scorecard at the end of this journey:
1.	[Data Preparation for PD Model](https://github.com/BAGLAT/Credit-Risk-Modeling/blob/master/Code/PD%20Model%20-%20Data%20Preparation%20(ipynb).ipynb)
2.	[PD Model Estimation and Validation](https://github.com/BAGLAT/Credit-Risk-Modeling/blob/master/Code/PD%20-%20Probability%20of%20Default%20Model%20Creation%20and%20Estimation%20(ipynb).ipynb)
3.	[LGD, EAD and Total Expected Loss](https://github.com/BAGLAT/Credit-Risk-Modeling/blob/master/Code/LGD%20%2B%20EAD%20Models%20and%20Final%20Total%20Expected%20Loss%20(ipynb).ipynb)

## Output
Given the funded amount our model predicts expected loss for each of the loan.

![alt]({{ site.url }}{{ site.baseurl }}/images/CRM/Output1.png)
![alt]({{ site.url }}{{ site.baseurl }}/images/CRM/Output2.png)

