# Databricks notebook source
# MAGIC %md
# MAGIC # **LOADING UP THE DATASET**
# MAGIC We started by loading up the dataset from the Unity catalog volume, and while loading, we made sure to account for the fact that the first row of the CSV file will contain the column names. We also accounted for fields that may contain line breaks. We made sure that if a field was enclosed in quotes and contained commas, spark would treat the quotes as the escape character, and we asked spark to detect the data type of each column. We read the CSV file from the given file into a dataframe that we titled df_raw. We did this using the spark.read function and then assigned options for the different things we were accounting for.

# COMMAND ----------

#Reading the dataset from Unity Catalog into a variable titled df_raw
df_raw = (
    spark.read
    .option("header", True)
    .option("multiLine", True)     # emails may contain line breaks
    .option("escape", '"')
    .option("inferSchema", True)
    .csv("/Volumes/teaching/datasets/assignment/emails.csv")
)

display(df_raw)

# COMMAND ----------

# MAGIC %md
# MAGIC # **CLEANING THE DATASET**
# MAGIC ## REMOVING NULL COLUMNS
# MAGIC Our first step for cleaning the data was to remove all rows where every column is null. This ensured that all useless rows were removed. We did this using the col function that we imported from pyspark.sql and ran the dropna command. We stored the changed data in a new dataframe called df.

# COMMAND ----------

#Cleaning the dataset
#Removing empty rows and standardizing column names
from pyspark.sql.functions import col

df = df_raw.dropna(how="all")

display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## EXTRACTING THE DATE PARSED COLUMN
# MAGIC Our next step in cleaning the data was to begin parsing relevant columns for the analysis we intend to do using the command withColumn. We started by parsing the dates that all emails were sent from the message column into a new column called date_parsed and then making the dates into a timestamp. This was done using the following functions from pyspark.sql: regexp_extract, regexp_replace, to_timestamp. We used the reg_extract function to look through the message column and extract any text that came after the string "Date" while also accounting for any spaces and we added the "1" specification to emsure that it only returns the first capture group which is the date string. We then used the reg_replace function to remove parenthesis and any text inside them example (PDT) and any spaces before the parenthesis, and also to remove the day of the week string at the beginning of the date string alongside its comma and the space after the comma, and this is to make sure that the date format will match the timestamp format that we need. We then used the to_timestamp function to convert it into a timestamp using the format "d MMM yyyy HH:mm:ss Z" which conforms with the datetime labelling for spark 3.0+

# COMMAND ----------

#Parsing the dates by extracting them from the message column into a new column titled date_parsed
from pyspark.sql.functions import regexp_extract, regexp_replace, to_timestamp

df = df.withColumn(
    "date_parsed",
    to_timestamp(
        regexp_replace(
            regexp_replace(
                regexp_extract(
                    col("message"),
                    r"Date:\s*(.*)",
                    1
                ),
                r"\s*\(.*\)", ""
            ),
            r"^[A-Za-z]+,\s*", ""
        ),
        "d MMM yyyy HH:mm:ss Z"
    )
)

display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## EXTRACTING "FROM_EMAIL" AND "TO_EMAIL" COLUMNS
# MAGIC The next 2 columns we created were the from_email and to_email columns and we did this using the following functions: regexp_extract, regexp_extract_all, explode, lit and the drop command. For the from_email column, We extracted it from the message column using a regex pattern that allowed us to search for and capture text after the string "From:" while accounting for any spaces in between and we gave it the specification "1" to ensure that they only returned the first capture group which is the email address. For the to_email column, we had a more difficult scenario because whilst there could only be one sender, there might be multiple recipients for any one mail. We tackled this by using the regexp_extract function to first extract the entire recipient line into a temporary column we called "to_line" using a regex pattern that captures everything after the string "to:" while accounting for spaces in between and for commas, and we gave it a specification of 1 to ensure that it returned the entire string. After this, we used the regexp_extract_all function to extract each individual email from the rows in "to_line" into an array of recipients using a regex pattern that we enclosed in the lit function so that it will be treated like a constant column, which captured each email address using the comma as the seperator. Unlike regexp_extract, there are no capture groups here so we gave it the specification "0" to ensure that it captures the entire match. Then, we used the explode function to seperate the arrays stored in the "to_emails" column into seperate rows all bearing the same sender but having just one recipient per row and we stored these new data in the "to_email" column. Finally, we used the drop command to get rid of the "to_line" and "to_emails" columns since they were no longer of use to us.

# COMMAND ----------

#Parsing the sender and recipient email addresses by extracting them from the message column into a new column titled from_email and to_email
from pyspark.sql.functions import regexp_extract_all, explode, lit
df = (
    df.withColumn(
        "from_email",
        regexp_extract(col("message"),
                       r"From:\s*([\w\.-]+@[\w\.-]+)",
                       1)
    )
    .withColumn(
        "to_line",
    regexp_extract(
        col("message"),
        r"To:\s*(.*?)(?:\n|$)",
        1
        )
    )
    .withColumn(
        "to_emails",
    regexp_extract_all(
        col("to_line"),
        lit(r"[\w\.-]+@[\w\.-]+"),
        0
        )
    )
    .withColumn("to_email", explode("to_emails"))
    .drop("to_line", "to_emails")
)
display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## EXTRACTING THE BODY COLUMN
# MAGIC The next column we extracted was the body column and we did this using the following functions: regexp_extract, regexp_replace, trim, and the drop command. To do this, we first used regexp_extract to extract the body from the message column using a regex pattern that captures everything after the line break which we discovered was used to seperate the header of the email from the body, and we stored this new data in a temporary column we called body_raw. We gave it a specification of "1" to ensure that it returned the first capture which is the entire body. After this, We needed to clean it by getting rid of any extra line breaks, tabs, multiple spaces, and any messy formatting. We did this by first using regexp_replace with a regex pattern to remove any whitespace characters which include spaces, tabs, newlines, and carriage returns, and replace them with a single space. Then, we used the trim function to remove any leading and trailing spaces, and then we saved the cleaned body in the column "body". After this, we used the drop command to remove the temporary column "body_raw" to prevent confusion as it was no longer needed.

# COMMAND ----------

#parsing the body of the email by extracting it from the message column into a new column titled body_raw and removing all whitespaces and spaces at the start and end of the string in another column titled body
from pyspark.sql.functions import trim

df = (
    df
    #Extracting body_raw
    .withColumn(
        "body_raw",
        regexp_extract(
            col("message"),
            r"\r?\n\r?\n([\s\S]*)",
            1
        )
    )
    #Cleaning body
    .withColumn(
        "body",
        trim(
            regexp_replace(col("body_raw"), r"\s+", " ")
        )
    )
    .drop("body_raw")
)

display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## EXTRACTING THE YEAR, MONTH, DAY AND HOUR COLUMNS
# MAGIC The next columns we created were the year, month, day and hour columns. We did this using the following functions: year, month, day and hour. We used these functions to extract the necessary data from the "date_parsed" column and stored them in the columns we created for them.

# COMMAND ----------

#Creating columns for the year, month, day of the week, and hour of the day of the emails
from pyspark.sql.functions import year, month, day, hour

df = (
    df.withColumn("year", year(col("date_parsed")))
      .withColumn("month", month(col("date_parsed")))
      .withColumn("day", day(col("date_parsed")))
      .withColumn("hour", hour(col("date_parsed")))
)

display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC # **QUESTION 1 (GROUP A)**
# MAGIC ## COUNTING TOTAL NUMBER OF UNIQUE RECIPIENTS FROM THE "TO_EMAIL" COLUMN
# MAGIC For Group A (easy). We chose to answer question 2 which asked the total number of unique recipients in the dataset (based on the "to_email" column). During the process of cleaning our data, we had already created this column and prepared the data in it. We simply made use of the filter command to remove any rows where the "to_email" column was empty or was displaying a null value, then we used the select command to make sure that spark was focused on the "to_email" column, then used the distinct command to remove any duplicates and keep only unique recipients, and finally used the count command to count the number of unique recipients and return a single number which we saved in a variable called unique_recipients. We then used the print function to display the result of the analysis.

# COMMAND ----------

#counting the number of unique recipients from the to_email column
unique_recipients = (
    df
    .filter(col("to_email").isNotNull() & (col("to_email") != ""))   #removing missing values and empty strings
    .select("to_email")
    .distinct()
    .count()
)

print("Total unique recipients:", unique_recipients)

# COMMAND ----------

# MAGIC %md
# MAGIC # **QUESTION 2 (GROUP B)**
# MAGIC ## COLLATING THE TOP 10 SENDERS BY EMAIL COUNT FROM THE "FROM_EMAIL" COLUMN
# MAGIC For Group B (medium). The first question we chose to answer is question 1 which asked for the top 10 senders by the number of emails sent(based on the "from_email" column), along with the number of emails they each sent. The functions we used to carry out this analysis were the col and count functions as well as the filter, groupby, agg(aggregate), alias, orderby, and limit commands. We started by removing any rows where the from_email is empty or displaying a null value to ensure that only valid sender addresses remain, then we used the groupby command to create one group per sender email, then used the agg command in addition to the count command to aggregate each group and provide a number of emails and used the alias command to rename the number of emails column to "emails_sent", we then used the orderby command to sort the column in descending order(from the largest to the smallest) and finally used the limit command to make sure that it only returns the first 10 rows when it is displayed. And this data was saved in a variable called top_senders.

# COMMAND ----------

#collating the top 10 senders by email count from the from_email column
from pyspark.sql.functions import count

top_senders = (
    df
    .filter(col("from_email").isNotNull() & (col("from_email") != ""))   #removing missing senders and empty strings
    .groupBy("from_email")
    .agg(count("*").alias("emails_sent"))
    .orderBy(col("emails_sent").desc())
    .limit(10)
)

display(top_senders)

# COMMAND ----------

# MAGIC %md
# MAGIC # **QUESTION 3 (GROUP B)**
# MAGIC ## COLLATING THE TOP 10 RECIPIENTS BY EMAIL COUNT FROM THE "TO_EMAIL" COLUMN
# MAGIC The second question we chose to answer was question 2 which asked the top 10 recipients by number of emails received (based on the to_email column) alongside the number of emails they received. We carried out a similar process to that of the sender question simply because we had already cleaned our data beforehand and created the to_email column.

# COMMAND ----------

#collating the top 10 recipients by email count from the to_email column
top_recipients = (
    df
    .filter(col("to_email").isNotNull() & (col("to_email") != ""))   #removing missing recipients and empty strings
    .groupBy("to_email")
    .agg(count("*").alias("emails_received"))
    .orderBy(col("emails_received").desc())
    .limit(10)
)

display(top_recipients)

# COMMAND ----------

# MAGIC %md
# MAGIC # **QUESTION 4 (GROUP C)**
# MAGIC ## COUNTING THE NUMBER OF EMAILS EXCHANGED IN 2001 AND VISUALISING THEM IN A LINE GRAPH
# MAGIC For Group C (difficult). We chose to answer question 3 which asked for the number of emails exchanged in the year 2001 and to also visualise this data in a line graph. The functions we used to do the analysis were: col, count, concat_ws and to_date as well as the filter, groupby, agg, alias, and orderby commands. The first thing we did was to filter out any emails not sent in the year 2001 and we did this by using the filter command on the "year" column that we created from the date_parsed (timestamp) column when cleaning our data, we then stored our filtered data in a variable called df_2001. The next step we took was to group and count the emails in the new variable and we did this by first using the groupby command in conjuction with the year, month and day columns to create groups for each calendar day, then we used the agg in addition to the count command to aggregate the rows in each group and return a count of each group and used the alias command to rename the column of count as emails_per_day, and finally we used the orderby command to sort the rows in chronological order starting from the first day of the year to the last day (which is essential for visualising our line graph). We stored this new data in a variable we named daily_counts. We still needed to compress the 3 columns we had for our date into a single column to allow us plot our visuals. We did this by creating a new column called date in our daily_counts variable which would contain the full date of each day and is a combination of the year, month and day columns. This was done using the concat_ws function which concatenates(combines) values with a seperator(-) and the to_date function which converts the concatenated string into a proper Date type. The final thing we did was to plot our line graph. We did this using the resources provided by Databricks, by clicking on the plus sign beside our displayed table and choosing visualisation, then choosing line chart, then setting our x-axis as date and our y-axis as emails_per_day and finally renaming the labellings to make them easy to read and understand.

# COMMAND ----------

from pyspark.sql.functions import concat_ws, to_date

#Filtering for 2001
df_2001 = df.filter(col("year") == 2001)

#Grouping by year, month, day and counting emails
daily_counts = (
    df_2001
    .groupBy("year", "month", "day")
    .agg(count("*").alias("emails_per_day"))
    .orderBy("year", "month", "day")
)

#Creating a proper date column for plotting
daily_counts = daily_counts.withColumn(
    "date", to_date(concat_ws("-", col("year"), col("month"), col("day")))
)

display(daily_counts)