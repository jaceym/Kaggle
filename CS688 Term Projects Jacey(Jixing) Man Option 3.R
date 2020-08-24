
#### CS688 Term Project Jixing Jacey Man
library(tm)
library(pdftools)
library(class)
library(cowplot)
library(rtweet)
library(ggplot2)
library(tidytext)
library(dplyr) 
library(SnowballC)
library(tidytext)
library(purrr)
library(tidyr)
library(corpus)
library(wordcloud)
library(RColorBrewer)
library(wordcloud2)

data("stop_words")
app_name = 'JM App to connect to R'
consumer_key = '9FoEz3PZwmKGhlNfRF3VGPbmI'
consumer_secret = 'bargI1DmyanBC9p7ASdedyf4Ea7dhiHJNV4QRU9GobhGL9rclW'
access_token = '872957667920281601-tRKN6HzzpeO9kqFrJglHq9G5R9lV6oM'
access_secret = 'gIjA0kpDwoDSZiHs416QD9Mb36i0KjyhrQe8Q1jvSVRDy'


create_token(app = app_name,
             consumer_key = consumer_key,
             consumer_secret = consumer_secret,
             access_token = access_token, 
             access_secret = access_secret)

### -- Pull Tweet content () --
### winner stock
set.seed(123)
uberSTK <- search_tweets("Uber Stock", n=100)
tslaSTK <- search_tweets("Tesla Stock", n=100)
fbSTK <- search_tweets("Facebook Stock", n=100)

### Combing Tweets for winner
winnerSTK <- rbind(uberSTK,tslaSTK,fbSTK)
winnerSTK
nrow(winnerSTK)

### losing stock
boeSTK <- search_tweets("Boeing Stock", n=100)
waySTK <- search_tweets("Wayfair Stock", n=100)
tmSTK <- search_tweets("3M Stock", n=100)

### Combing Tweets for loser
loserSTK <- rbind(boeSTK,waySTK,tmSTK)
loserSTK
nrow(loserSTK)
set.seed(123)

#### Part B
set.seed(123)
WT <- tibble(winnerSTK$text)
head(WT)
LT <- tibble(loserSTK$text)
head(LT)
dfWCorpus = Corpus(VectorSource(WT)) 
dfLCorpus = Corpus(VectorSource(LT)) 

#### Part C
set.seed(123)
transformW <- tm_map(dfWCorpus, content_transformer(tolower))
transformW[[1]]$content
transformW1 <- tm_map(transformW, removeWords, stopwords("english"))
transformW1[[1]]$content
transformW2 <- tm_map(transformW1, removePunctuation)
transformW2[[1]]$content
transformW3 <- tm_map(transformW2, removeWords, letters)
transformW3[[1]]$content

transformL <- tm_map(dfLCorpus, content_transformer(tolower))
transformL[[1]]$content
transformL1 <- tm_map(transformL, removeWords, stopwords("english"))
transformL1[[1]]$content
transformL2 <- tm_map(transformL1, removePunctuation)
transformL2[[1]]$content
transformL3 <- tm_map(transformL2, removeWords, letters)
transformL3[[1]]$content


### Part D
dtm1 <- DocumentTermMatrix(transformW3)
inspect(dtm1)
dtm1
dtm2 <- DocumentTermMatrix(transformL3)
inspect(dtm2)

### Part E 

freq <- colSums(as.matrix(dtm1)) # calculate the frequency of each term in the DTM
ord <- order(freq, decreasing = TRUE) # sort the freq list in descending order
x <- as.data.frame(freq[head(ord, n=20)]) # get the first 50 most frequent terms from the freq list
x

freq2 <- colSums(as.matrix(dtm2)) # calculate the frequency of each term in the DTM
ord2 <- order(freq2, decreasing = TRUE) # sort the freq list in descending order
y <- as.data.frame(freq2[head(ord2,n=20)]) # get the first 50 most frequent terms from the freq list
y

set.seed(999)
wordcloud(row.names(x), scale=c(2,0.8), colors=brewer.pal(6, "Dark2"))

set.seed(999)
wordcloud(row.names(y), scale=c(2,0.8), colors=brewer.pal(6, "Dark2"))


### Part F
### -- Sentiment analysis --

winner_words <- tibble(word = c("https","t.co","tesla","stock","facebook","uber"))
loser_words <- tibble(word = c("https","t.co","boeing","stock","wayfair","3m"))

get_sentiments('nrc')
get_sentiments('bing')

bing_winner <- winnerSTK[,5] %>%
  unnest_tokens(word, text) %>%
  anti_join(stop_words, by = c('word')) %>%
  anti_join(winner_words, by = c('word')) %>%
  inner_join(get_sentiments("bing")) %>%
  count(word, sentiment, sort=TRUE) %>%
  ungroup()
bing_winner

bing_loser <- loserSTK[,5] %>%
  unnest_tokens(word, text) %>%
  anti_join(stop_words, by = c('word')) %>%
  anti_join(loser_words, by = c('word')) %>%
  inner_join(get_sentiments("bing")) %>%
  count(word, sentiment, sort=TRUE) %>%
  ungroup()
bing_loser

bing_winner %>%
  group_by(sentiment) %>%
  top_n(10) %>%
  ungroup() %>%
  mutate(word = reorder(word, n)) %>%
  ggplot(aes(word, n, fill = sentiment)) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~sentiment, scales = "free_y") +
  labs(title = "Winner Stock",
       y = "Contribution to sentiment",
       x = NULL) +
  coord_flip() + theme_bw()

bing_loser %>%
  group_by(sentiment) %>%
  top_n(10) %>%
  ungroup() %>%
  mutate(word = reorder(word, n)) %>%
  ggplot(aes(word, n, fill = sentiment)) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~sentiment, scales = "free_y") +
  labs(title = "Loser Stock",
       y = "Contribution to sentiment",
       x = NULL) +
  coord_flip() + theme_bw()


### Extra Credit

library(quantmod)
library(lubridate)
library(ggplot2)
library(cowplot)


MMM <- getSymbols("MMM", auto.assign=FALSE)
MMM.matrix <- as.matrix(MMM)
times <- ymd(rownames(MMM.matrix))
MMM.df <- data.frame(date=times,price=MMM.matrix[,"MMM.Adjusted"])
MMM.df.since2020 <- MMM.df[MMM.df$date>"2020-03-28",]
p1 <- ggplot(data=MMM.df.since2020,aes(x=date,y=price)) + geom_line() + labs(title="Change in 3M Stocks 3-28")

UBER<- getSymbols("UBER", auto.assign=FALSE)
UBER.matrix <- as.matrix(UBER)
times <- ymd(rownames(UBER.matrix))
UBER.df <- data.frame(date=times,price=UBER.matrix[,"UBER.Adjusted"])
UBER.df.since2020 <- UBER.df[UBER.df$date>"2020-03-28",]
p4 <- ggplot(data=UBER.df.since2020,aes(x=date,y=price)) + geom_line() + labs(title="Change in Uber Stocks 3-28")

FB <- getSymbols("FB", auto.assign=FALSE)
FB.matrix <- as.matrix(FB)
times <- ymd(rownames(FB.matrix))
FB.df <- data.frame(date=times,price=FB.matrix[,"FB.Adjusted"])
FB.df.since2020 <- FB.df[FB.df$date>"2020-03-28",]
p5 <- ggplot(data=FB.df.since2020,aes(x=date,y=price)) + geom_line() + labs(title="Change in Facebook Stocks 3-28")

TSLA <- getSymbols("TSLA", auto.assign=FALSE)
TSLA.matrix <- as.matrix(TSLA)
times <- ymd(rownames(TSLA.matrix))
TSLA.df <- data.frame(date=times,price=TSLA.matrix[,"TSLA.Adjusted"])
TSLA.df.since2020 <- TSLA.df[TSLA.df$date>"2020-03-28",]
p6 <- ggplot(data=TSLA.df.since2020,aes(x=date,y=price)) + geom_line() + labs(title="Change in Tesla Stocks 3-28")

BA <- getSymbols("BA", auto.assign=FALSE)
BA.matrix <- as.matrix(BA)
times <- ymd(rownames(BA.matrix))
BA.df <- data.frame(date=times,price=BA.matrix[,"BA.Adjusted"])
BA.df.since2020 <- BA.df[BA.df$date>"2020-03-28",]
p2 <- ggplot(data=BA.df.since2020,aes(x=date,y=price)) + geom_line() + labs(title="Change in Boeing Stocks 3-28")

W <- getSymbols("W", auto.assign=FALSE)
W.matrix <- as.matrix(W)
times <- ymd(rownames(W.matrix))
W.df <- data.frame(date=times,price=W.matrix[,"W.Adjusted"])
W.df.since2020 <- W.df[W.df$date>"2020-03-28",]
p3 <- ggplot(data=W.df.since2020,aes(x=date,y=price)) + geom_line() + labs(title="Change in Wayfair Stocks 3-28")

plot_grid(p1, p2, p3, p4,p5,p6)
