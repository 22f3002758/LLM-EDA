# Happiness Metrics Over Time

This dataset appears to track various metrics of happiness and well-being across different countries and years, highlighting factors like economic performance, social support, health, and subjective well-being.

## Dataset Summary
- **Dataset Size**: (2363, 11)
- **Column Names**: 
  - ['Country name', 'year', 'Life Ladder', 'Log GDP per capita', 'Social support', 'Healthy life expectancy at birth', 'Freedom to make life choices', 'Generosity', 'Perceptions of corruption', 'Positive affect', 'Negative affect']

### Numerical Summary Statistics

|          | year          | Life Ladder | Log GDP per capita | Social support | Generosity | Perceptions of corruption | Positive affect | Negative affect |
|----------|---------------|-------------|--------------------|----------------|------------|--------------------------|-----------------|-----------------|
| count    | 2363.000000   | 2363.000000 | 2363.000000        | 2363.000000    | 2282.000000| 2238.000000              | 2339.000000     | 2347.000000     |
| mean     | 2014.763860   | 5.483566    | 9.399671           | 0.809369       | 0.000098   | 0.743971                 | 0.651882        | 0.273151        |
| std      | 5.059436      | 1.125522    | 1.145221           | 0.120878       | 0.161388   | 0.184865                 | 0.106240        | 0.087131        |
| min      | 2005.000000   | 1.281000    | 5.527000           | 0.228000       | -0.340000  | 0.035000                 | 0.179000        | 0.083000        |
| 25%      | 2011.000000   | 4.647000    | 8.520000           | 0.744000       | -0.112000  | 0.687000                 | 0.572000        | 0.209000        |
| 50%      | 2015.000000   | 5.449000    | 9.492000           | 0.834000       | -0.022000  | 0.798500                 | 0.663000        | 0.262000        |
| 75%      | 2019.000000   | 6.323500    | 10.382000          | 0.904000       | 0.093750   | 0.867750                 | 0.737000        | 0.326000        |
| max      | 2023.000000   | 8.019000    | 11.676000          | 0.987000       | 0.700000   | 0.983000                 | 0.884000        | 0.705000        |

### EDA

#### Insights from Numerical Heatmap Analysis
Oh, what an intricate tapestry we have here with this numerical heatmap! Let’s dive into some fascinating insights.

1. **Life Ladder and Log GDP per capita**: There's a robust correlation of 0.77 here! This suggests that as the GDP per capita increases, so does the perceived quality of life—a clear indication that economic prosperity positively impacts happiness.
2. **Social Support**: It strikes at the heart of community bonds, with a strong correlation of 0.67 to the Life Ladder. This finding hints that people who feel supported by their community tend to report higher life satisfaction.
3. **Healthy Life Expectancy at Birth**: This metric shares a commendable correlation of 0.66 with the Life Ladder, underscoring the importance of health in determining overall happiness. It serves as a reminder that a longer, healthier life can lead to broader life satisfaction.
4. **Log GDP per capita and Social Support**: The lower correlation of 0.28 suggests a more complex relationship, indicating that wealth alone doesn’t always translate directly to community support.
5. **Yearly Trends**: The analysis reveals a somewhat unexpected correlation of -0.043 between year and Social Support, signaling that over the years, perceptions of social support might not be simply improving alongside economic conditions.

In summary, this heatmap paints a colorful picture of how various factors intertwine to shape our perceptions of quality of life. Each correlation tells a story of how our financial, social, and health environments influence our overall happiness!

## Key Analysis
1. **Yearly Trends in Life Ladder**
   ![yearly_trends_life_ladder.png](image_name)
   - The chart depicting the Yearly Trends in Life Ladder scores paints a fascinating picture of life satisfaction dynamics over time. Notably, there’s a significant dip in 2006, where scores plummeted to around 5.2, suggesting a period of major upheaval or discontent. Following this, we observe a generally upward trend, with scores gradually climbing back up, only to stabilize around 5.6 in recent years.

2. **Correlation Analysis of Economic Indicators**
   ![correlation_analysis_economic_indicators.png](image_name)
   - This correlation matrix reveals fascinating relationships among three key economic metrics: Log GDP per capita, Social support, and Life Ladder. Notably, Log GDP per capita shares a robust correlation with Life Ladder (0.77), suggesting that as economic prosperity increases, individuals tend to report higher satisfaction with their lives.

3. **Top N Countries by Life Ladder**
   ![top_n_countries_life_ladder.png](image_name)
   - In our vibrant exploration of the 'Top 10 Countries by Life Ladder Score,' we uncover a truly fascinating tableau of human well-being! Leading the pack, Denmark stands tall, followed closely by Finland and Iceland, showcasing the Nordic countries' remarkable commitment to quality of life.

4. **Distribution of Healthy Life Expectancy**
   ![distribution_healthy_life_expectancy.png](image_name)
   - This box plot provides a captivating glimpse into the world of healthy life expectancy at birth across various nations. The central box indicates that most countries enjoy a healthy life expectancy from around 60 to 70 years, showcasing a positive public health trend.

### Summary
The analysis of happiness metrics over time showcases the intricate relationships between economic indicators, social support, and subjective well-being across different countries. Key trends highlight the complexities of life satisfaction, emphasizing both economic prosperity and community bonds as critical factors influencing happiness.