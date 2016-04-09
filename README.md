Shared Task Intro
------------

distribution of relations in the training set.

```shell 
Counter({u'Comparison': 492,
         u'Comparison.Concession': 1277,
         u'Comparison.Contrast': 4602,
         u'Contingency': 2,
         u'Contingency.Cause': 1,
         u'Contingency.Cause.Reason': 3078,
         u'Contingency.Cause.Result': 2006,
         u'Contingency.Condition': 1152,
         u'EntRel': 4133,
         u'Expansion': 99,
         u'Expansion.Alternative': 206,
         u'Expansion.Alternative.Chosen alternative': 238,
         u'Expansion.Conjunction': 7644,
         u'Expansion.Exception': 15,
         u'Expansion.Instantiation': 1401,
         u'Expansion.Restatement': 2664,
         u'Temporal': 5,
         u'Temporal.Asynchronous': 3,
         u'Temporal.Asynchronous.Precedence': 1230,
         u'Temporal.Asynchronous.Succession': 985,
         u'Temporal.Synchrony': 1302})
```

![alt tag](https://raw.githubusercontent.com/SBelkaid/NLPToolkits/master/images/Screen%20Shot%202016-04-09%20at%208.19.54%20PM.png)




Installation
-----------
Clone the repository from github

````shell
git clone https://github.com/SBelkaid/NLPToolkits.git
```` 


Usage
-----

This is a script to explore the data, feel free to change anything you want.
Change the USE variable to one of the two variables above to change the path to the data you want to use. Two paths are given of which the first is the trial data provided by the tutorial and the second is the path to the actual train data. Make sure that you put the folder in the same relative path as specified in the script. 

```shell

NameOfComputer: python exploratory_conll.py

```