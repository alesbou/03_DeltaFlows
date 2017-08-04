library(ggplot2)
library(ggjoy)
library(hrbrthemes)
library(viridis)
library(tidyverse)
library(forcats)
library(grid)
library(gridExtra)

setwd("/Users/Alvar/Desktop/coding/03_DeltaFlows/")
d <- read.csv(file="monthsummary.csv", header=TRUE, sep=",")
d <- subset(d, Year>1994)
d$MonthNumber <- as.factor(d$Month)

ggplot(d, aes(x = EXPORT, y = MonthNumber,fill = ..y..)) +
  geom_joy2(scale=2.5)+
  scale_x_continuous(limits=c(0, 1000))+
  scale_y_discrete(labels=month.abb)+
  scale_fill_viridis()+
  labs(x = "Exports (taf/month)",
       y = "",
       title = "Historic Water Exports from the Delta by Month",
       subtitle="Data from Dayflow, DWR (1995-2016)",
       caption = "Alvar Escriva-Bou (@AlvarEscriva)") +
  theme_joy(font_size = 13,grid = FALSE)+
  theme(legend.position="none")
