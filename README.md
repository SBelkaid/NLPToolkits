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

Output of comparing the salience using tfidf to calculate the salience of both arguments. Usually resulting in a higher salience for the fir argument.
 
```shell
ARG1: Solo woodwind players have to be creative
SALIENCE ARG1 1.96610017517
CONECTIVE: IF
SENSE: Contingency.Condition
ARG2: they want to work a lot
SALIENCE ARG2 1.72814871799 


ARG1: Solo woodwind players have to be creative if they want to work a lot
SALIENCE ARG1 2.50204455491
CONECTIVE: BECAUSE
SENSE: Contingency.Cause.Reason
ARG2: their repertoire and audience appeal are limited
SALIENCE ARG2 1.92700701696 


ARG1: He commissions and splendidly interprets fearsome contemporary scores and does some conducting
SALIENCE ARG1 2.76420720709
CONECTIVE: SO
SENSE: Contingency.Cause.Result
ARG2: he doesn't have to play the same Mozart and Strauss concertos over and over again
SALIENCE ARG2 2.12111288707 


ARG1: Today, the pixie-like clarinetist has mostly dropped the missionary work
SALIENCE ARG1 2.40192146751
CONECTIVE: THOUGH
SENSE: Comparison.Concession
ARG2: a touch of the old Tashi still survives
SALIENCE ARG2 1.9027858474 


ARG1: Just the thing for the Vivaldi-at-brunch set, the yuppie audience that has embraced New Age as its very own easy listening
SALIENCE ARG1 3.24105620405
CONECTIVE: BUT
SENSE: Comparison.Concession
ARG2: you can't dismiss Mr. Stoltzman's music or his motives as merely commercial and lightweight
SALIENCE ARG2 2.67478028335 


ARG1: He believes in what he plays
SALIENCE ARG1 1.40134849657
CONECTIVE: AND
SENSE: Expansion.Conjunction
ARG2: he plays superbly
SALIENCE ARG2 1.39120620883 


ARG1: that his new album, "Inner Voices," had just been released, that his family was in the front row
SALIENCE ARG1 2.65605945037
CONECTIVE: AND
SENSE: Expansion.Conjunction
ARG2: that it was his mother's birthday
SALIENCE ARG2 1.38858359134 


ARG1: and that it was his mother's birthday
SALIENCE ARG1 1.38858359134
CONECTIVE: SO
SENSE: Contingency.Cause.Result
ARG2: he was going to play her favorite tune from the record
SALIENCE ARG2 2.15514823413 


ARG1: He launched into Saint-Saens's "The Swan" from "Carnival of the Animals," a favorite encore piece for cellists, with lovely, glossy tone and no bite
SALIENCE ARG1 3.68486563828
CONECTIVE: THEN
SENSE: Temporal.Asynchronous.Precedence
ARG2: he offered the second movement from Saint-Saens's Sonata for Clarinet, a whimsical, puckish tidbit that reflected the flip side of the Stoltzman personality
SALIENCE ARG2 3.6259050031 


ARG1: Then he offered the second movement from Saint-Saens's Sonata for Clarinet, a whimsical, puckish tidbit that reflected the flip side of the Stoltzman personality
SALIENCE ARG1 3.6259050031
CONECTIVE: AS IF
SENSE: Expansion
ARG2: to show that he could play fast as well
SALIENCE ARG2 1.41419501089

```

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