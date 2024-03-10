# -----------------------------------------------------------------
# Structural Topic Modeling 
# Jovan Trajceski
# -----------------------------------------------------------------

# Clear up data in global environment
rm(list=ls())

# Run and load Libraries
library(topicmodels)
library(lda)
library(slam)
library(stm)
library(ggplot2)
library(dplyr)
library(tidytext)
# try to make it faster
library(furrr)
plan(multicore)
library(tm) # Framework for text mining
library(tidyverse) # Data preparation and pipes %>%
library(ggplot2) # For plotting word frequencies
library(wordcloud) # Wordclouds!
library(Rtsne)
library(rsvd)
library(geometry)
library(NLP)
library(ldatuning) 


# Load data from csv file
news <- read.csv("../../data/art_dataset_with_meta.csv")
#news <- read.csv('raw_partner_headlines.csv')
# Check for NAs - no NAs
sapply(news, function(x) sum(is.na(x)))

# Overview of original dataset
str(news)
sapply(news, typeof)


# randomly sample 1000 rows + remove unnecessary columns
set.seed(830)
#news_sample <-news[sample(nrow(news), 1000), -c(2,3)]
news_sample <-news[-c(3,4,5)]

# Format and transform columns
#news_sample$date <- as.Date(news_sample$date)
#news_sample$date <- strftime(news_sample$date, format = "%j")
#news_sample$date <- as.integer(news_sample$date) 

news_sample$f0 <- as.factor(news_sample$f0)
news_sample$f1 <- as.factor(news_sample$f1)
news_sample$f2 <- as.factor(news_sample$f2)
news_sample$f3 <- as.factor(news_sample$f3)
news_sample$f4 <- as.factor(news_sample$f4)
news_sample$f5 <- as.factor(news_sample$f5)
news_sample$f6 <- as.factor(news_sample$f6)
news_sample$f7 <- as.factor(news_sample$f7)
news_sample$f8 <- as.factor(news_sample$f8)
news_sample$f9 <- as.factor(news_sample$f9)
news_sample$f10 <- as.factor(news_sample$f10)
news_sample$f11 <- as.factor(news_sample$f11)
news_sample$f12 <- as.factor(news_sample$f12)
news_sample$f13 <- as.factor(news_sample$f13)
news_sample$f14 <- as.factor(news_sample$f14)
news_sample$f15 <- as.factor(news_sample$f15)
news_sample$f16 <- as.factor(news_sample$f16)
news_sample$f17 <- as.factor(news_sample$f17)
news_sample$f18 <- as.factor(news_sample$f18)
news_sample$f19 <- as.factor(news_sample$f19)
news_sample$f20 <- as.factor(news_sample$f20)
news_sample$f21 <- as.factor(news_sample$f21)
news_sample$f22 <- as.factor(news_sample$f22)
news_sample$f23 <- as.factor(news_sample$f23)
news_sample$f24 <- as.factor(news_sample$f24)
news_sample$f25 <- as.factor(news_sample$f25)
news_sample$f26 <- as.factor(news_sample$f26)
news_sample$f27 <- as.factor(news_sample$f27)
news_sample$f28 <- as.factor(news_sample$f28)
news_sample$f29 <- as.factor(news_sample$f29)
#news_sample$stock <- as.factor(news_sample$stock)

# Double-check format
sapply(news_sample, typeof)

# Pre-processing within the stm package before we run the topic meodel
# More info at https://search.r-project.org/CRAN/refmans/stm/html/textProcessor.html
# The stm package converts a vector of text and a dataframe of metadata into stm formatted objects 
# using the command textProcessor which calls the package tm for its pre-processing routines.
# * default parameters

processed <- textProcessor(news_sample$pre_text, metadata = news_sample,
                           lowercase = FALSE, #*
                           removestopwords = FALSE, #*
                           removenumbers = FALSE, #*
                           removepunctuation = FALSE, #*
                           stem = FALSE, #*
                           wordLengths = c(3,Inf), #*
                           sparselevel = 1, #*
                           language = "fa", #*
                           verbose = TRUE, #*
                           onlycharacter = TRUE, # not def
                           striphtml = FALSE, #*
                           customstopwords = NULL, #*
                           v1 = FALSE) #*

# The processed object is a list of four objects: documents, vocab, meta, and docs.removed. The documents 
# object is a list, one per document, of 2 row matrices; the first row indicates the index of a word found 
# in the document, and the second row indicates the (nonzero) counts. If preprocessing causes any documents 
# to be empty, they are removed, as are the corresponding rows of the meta object.

# filter out terms that don’t appear in more than 10 documents,
out <- prepDocuments(processed$documents, processed$vocab, processed$meta, lower.thresh=10)
docs <- out$documents
vocab <- out$vocab
meta <-out$meta

# Check levels
levels(meta$f1)

# Run initial topic model at 15 topics and see how long it takes
# Run time: 2 seconds on i7 CPU (12 cores)
set.seed(831)
system.time({
  First_STM <- stm(docs, vocab, 3,
                   prevalence =~ f0+f1+f2+f3+f4+f5+f6+f7+f8+f9,
                   data = meta,
                   seed = 15, max.em.its = 75
  )
})

# Plot first Topic Model
Sys.setlocale(locale = "Persian")
windowsFonts(A = windowsFont("B Nazanin"))
op=par(family="A", font.lab=2)
par(op)


labelTopics(First_STM)
news$pred <- First_STM$theta

write.csv(news, "../../data/art_dataset-stm_pred.csv")


plot(First_STM)
dev.off()  

# Let’s see what our model came up with! The following tools can be used to evaluate the model:
# labelTopics gives the top words for each topic.
# findThoughts gives the top documents for each topic (the documents with the highest proportion of each topic).


# Top Words
labelTopics(First_STM)

# We can find the top documents associated with a topic with the findThoughts function:
# top 2 paragraps for Topic #1 to 10
findThoughts(Third_STM, texts = meta$headline,n = 2, topics = 1:10)

# We can look at multiple, or all, topics this way as well. For this we’ll just look at the shorttext.
# top 3 paragraps for Topic #1 to 15
findThoughts(Third_STM, texts = meta$headline,n = 3, topics = 1:15)


# Graphical display of topic correlations
topic_correlation<-topicCorr(Third_STM)
plot(topic_correlation)


# Graphical display of convergence
plot(Third_STM$convergence$bound, type = "l",
     ylab = "Approximate Objective",
     main = "Convergence")



# Wordcloud:topic 17 with word distribution
set.seed(837)
cloud(Third_STM, topic=17, scale=c(10,2))


# Working with meta-data 
# Change topics # from 1:10 or larger
set.seed(837)
predict_topics<-estimateEffect(formula = 1:10 ~ publisher + s(date), 
                               stmobj = Third_STM, 
                               metadata = out$meta, 
                               uncertainty = "Global",
                               prior = 1e-5) # Adding a small prior 1e-5 for numerical stability.

# Effect of Zacks vs . Seeking Alpha publishers
set.seed(837)
plot(predict_topics, covariate = "publisher", topics = c(1,4,10),
     model = Third_STM, method = "difference",
     cov.value1 = "Zacks", cov.value2 = "Seeking Alpha",
     xlab = "More Seeking Alpha ... More Zacks",
     main = "Effect of Zacks vs. Seeking Alpha",
     xlim = c(-.1, .1), labeltype = "custom",
     custom.labels = c('Topic 1','Topic 4','Topic 10'))


# Effect of 'TalkMarkets' vs. 'Investopedia' publishers
set.seed(837)
plot(predict_topics, covariate = "publisher", topics = c(1,4,10),
     model = Third_STM, method = "difference",
     cov.value1 = "TalkMarkets", cov.value2 = "Investopedia",
     xlab = "More Investopedia ... More TalkMarkets",
     main = "Effect of TalkMarkets vs. Investopedia",
     xlim = c(-.1, .1), labeltype = "custom",
     custom.labels = c('Topic 1','Topic 4','Topic 10'))



# We can use plot() and type = perspectives to compare two topics or a single topic across 
# two covariate levels to see how the terms differ. 
# We use set.seed() to make the output reproducible. Comparing the content in two topics
set.seed(831)

plot(Third_STM, 
     type="perspectives", 
     topics=c(17,12), 
     plabels = c("Topic 17","Topic 12"))


# Topic proportions within documents for 9 topics 
plot(First_STM, type = "hist", topics = sample(1:20, size = 3))

plot(First_STM, type="hist")


# The topicQuality() function plots these values and labels each with its topic number:
topicQuality(model=First_STM, documents=docs)


# This code is free to use for academic purposes only, provided that a proper reference is cited. 
# This code comes without technical support of any kind. 
# Under no circumstances will the author be held responsible for any use of this code in any way.

