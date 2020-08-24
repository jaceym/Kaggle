# Kaggle
Twitter stock market sentiment analysis term project: (detailed code please see full R file)
1. Dataset: I picked 6 stocks, 
Winner stocks in my case are: Facebook, Uber, Tesla
Loser stocks in my case are: 3M, Wayfair, Boeing

2. The way I decided to search for the symbol is “Company name + Stock”
Example: (see full code for detail)
uberSTK <- search_tweets("Uber Stock", n=100)
tslaSTK <- search_tweets("Tesla Stock", n=100)
fbSTK <- search_tweets("Facebook Stock", n=100)
The reason why is because usually when people discuss stock, they would put company name and the word “stock” together, rather than just the symbol or the company name itself. Not everyone knows the exact symbol, and if they use just the company name, they could refer to some other topic related to the company, not just stocks. 
	Use the below code, I combined the 6 stock into two groups:
winnerSTK <- rbind(uberSTK,tslaSTK,fbSTK)
winnerSTK
loserSTK <- rbind(boeSTK,waySTK,tmSTK)
loserSTK
I use “rbind” since the output of tweet search are in rows, so I just aggregated the tweets out put row together.

3. Create two separate data corpora (or tidy text objects) for the above two sets of tweets. 
WT <- tibble(winnerSTK$text)
head(WT)
LT <- tibble(loserSTK$text)
head(LT)
dfWCorpus = Corpus(VectorSource(WT)) 
dfLCorpus = Corpus(VectorSource(LT))
I used the above code to create corpus, since I only want the tweet text of the tweet search output, I used $text to get just that column from my tweet search results.

4. Use the necessary pre-processing transformations described in the lecture notes. 
I used some common pre-processing such as change to lower case, remove English stop words and remove punctuation to produce better text dataset. (see full R code for detail)
transformW <- tm_map(dfWCorpus, content_transformer(tolower))
transformW[[1]]$content
transformW1 <- tm_map(transformW, removeWords, stopwords("english"))
transformW1[[1]]$content
transformW2 <- tm_map(transformW1, removePunctuation)
transformW2[[1]]$content
transformW3 <- tm_map(transformW2, removeWords, letters)
transformW3[[1]]$content


5. Create the document-term matrix for each set. Name them dtm1 and dtm2. 
Inspecting DTM file, making sure I am getting the right data. 
dtm1 <- DocumentTermMatrix(transformW3)
inspect(dtm1)
dtm1
dtm2 <- DocumentTermMatrix(transformL3)
inspect(dtm2)
(see below for inspect sample output)
 
6. Find the most frequent terms from each set. 
	This is the most frequent term for winner set
 
	This is the most frequent term for loser set
 
7. Show a word cloud for each set. 
This is the word cloud for winner set
 
This is the world cloud for loser set
 
8. Using the positive and negative word lists, compute the sentiment score (as described in the lecture) for all the tweets for each gainers (losers) set. Were the tweets about the 3 largest gainer stocks for that day characterized by a positive sentiment, and the tweets about the 3 largest loser stocks for that day characterized by a negative sentiment? 
These are list of winner stock term and their scores
 
These are list of loser stock term and their scores
 
Below are the charts for both group, it is very clear on the sentiment chart that the loser group has more term in the negative spectrum than the winner stock group:
1.	Winner stock sentiment chart
 
2.	Loser stock sentiment chart
 
9. Extra Credit: create ONE appropriate data visualization, using the principles of Module 6, which shows the stock prices and/or the change in stock prices for the stocks and day you selected for this project. 
Below are the a group chart of all 6 stocks, the 3 chart on the first row are loser stocks, the 3 chart on the second row are winner stocks. All winner stocks are all going in up-ward trend, while on the loser stock, Boeing has dropped dramatically. Also the loser stocks has more up and downs.

 
