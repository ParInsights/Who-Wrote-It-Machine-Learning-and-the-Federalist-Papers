# Who Wrote It? 
# Using Machine Learning to Determine Who Wrote the Disputed Essays of the Federalist Papers.
## Was it Madison or Hamilton?

We will also try to see which algorithms yield the most accurate results? 

Method 1 involves using clustering algorithms - k-Means, EM, and HAC.
Method 2 involves classification algorithms - deicison trees 

### 1. About the Federalist Papers
Quote from the Library of Congress http://www.loc.gov/rr/program/bib/ourdocs/federalist.html

The Federalist Papers were a series of eighty-five essays urging the citizens of New York to ratify the new United States Constitution. Written by Alexander Hamilton, James Madison, and John Jay, the essays originally appeared anonymously in New York newspapers in 1787 and 1788 under the pen name "Publius." A bound edition of the essays was first published in 1788, but it was not until the 1818 edition published by the printer Jacob Gideon that the authors of each essay were identified by name. The Federalist Papers are considered one of the most important sources for interpreting and understanding the original intent of the Constitution.

### 2. About the disputed authorship
The original essays can be downloaded from the Library of Congress. http://thomas.loc.gov/home/histdox/fedpapers.html

In the author column, you will find 74 essays with identified authors: 51 essays written by Hamilton, 15 by Madison, 3 by Hamilton and Madison, 5 by Jay. The remaining 11 essays, however, is authored by “Hamilton or Madison”. These are the famous essays with disputed authorship. Hamilton wrote to claim the authorship before he was killed in a duel. Later Madison also claimed authorship. Historians were trying to find out which one was the real author.


### 3. Computational approach for authorship attribution
In 1960s, statistician Mosteller and Wallace analyzed the frequency distributions of common function words in the Federalist Papers, and drew their conclusions. This is a pioneering work on using mathematical approaches for authorship attribution. http://www.stat.cmu.edu/~vlachos/courses/724/final/mosteller.pdf

Nowadays, authorship attribution has become a classic problem in the data mining field, with applications in forensics (e.g. deception detection), and information organization.


### The Goal of This Project
Is to provide evidence for each method and to demonstrate what patterns have been learned to predict the disputed papers. Each method has its own report. 
