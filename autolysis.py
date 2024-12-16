# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "httpx",
#   "pandas",
# "seaborn",
#  "scipy",
#  "numpy",
#  "scikit-learn",
#  "matplotlib",
#  "requests",
#  "chardet"
# ]
# ///

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
import seaborn as sns
import os
import chardet
import codecs
import sys
import warnings
import traceback
import shutil
import json
import base64
warnings.filterwarnings('ignore')

AIPROXY_TOKEN = os.environ["AIPROXY_TOKEN"]

def query_llm(data):
    """
    Sends a query to the LLM using the given data.

    Parameters:
        data (dict): The payload containing the model, messages, and function call details.

    Returns:
        dict: The function call arguments returned by the LLM.

    Raises:
        Exception: If the request fails or the response format is invalid.
    """
    try:
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {AIPROXY_TOKEN}"}
        url = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"
        response = requests.post(url=url, headers=headers, json=data)
        response.raise_for_status()
        return response.json()['choices'][0]['message']['function_call']['arguments']
    except Exception as e:
        print("Error querying the LLM:", e)
        raise

def validate_column(df, i_c):
    """
    Validates columns in a DataFrame for data type consistency, missing values, and correctness.

    Parameters:
        df (pd.DataFrame): The DataFrame to validate.
        i_c (list): A list of column names to validate.

    Returns:
        dict: The validation analysis results for each column.
    """
    try:
        data = {}
        missing_cells = {}
        current_dtype = {}

        for column in i_c:
            data[column] = df[column].head(10).tolist()
            missing_cells[column] = df[column].isnull().sum()
            current_dtype[column] = df[column].dtype

        sample_data = [
            {"column_name": column, "sample_data": data[column], "missing_cells": missing_cells[column]}
            for column in i_c
        ]

        json_data = {
            "model": "gpt-4o-mini",
            "messages": [
                {
                    "role": "system",
                    "content": "You are a dataset analysis assistant. Provide structured and concise output using the 'validate_column_data' function."
                },
                {
                    "role": "user",
                    "content": f'''Analyze the data of specific columns from the dataset based on their provided sample values and the current data type. For each column:
                    1. Determine the correct data type based on the sample data and column name. Options: 'integer', 'float', 'string', 'boolean', or 'date'.
                    2. If the current data type in the dataset differs from the inferred type, explain why this discrepancy may exist.
                    3. Suggest a resolution to address the discrepancy and ensure data consistency.
                    4. Indicate 'Yes' for action needed if the current data type differs from the inferred data type or there are missing/invalid values, otherwise 'No'.
                    Strictly base your analysis on the provided data.
                    Here are the columns to analyze:
                    - Column name and sample data: {sample_data}
                    - Current data type: {current_dtype}'''
                }
            ],
            "functions": [
                {
                    "name": "validate_column_data",
                    "description": "Validate the data type and consistency of specific columns in a dataset, identify discrepancies, and suggest resolutions.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "column_analysis": {
                                "type": "array",
                                "description": "List of columns analyzed with inferred data types, reasons for mismatched data types, and suggested resolutions.",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "column_name": {
                                            "type": "string",
                                            "description": "The name of the column being analyzed."
                                        },
                                        "inferred_data_type": {
                                            "type": "string",
                                            "description": "The correct data type of the column based on the sample data. Options include 'integer', 'float', 'string', 'boolean', or 'date'."
                                        },
                                        "reason_for_discrepancy": {
                                            "type": "string",
                                            "description": "Explanation of why the current data type differs from the inferred type, if applicable. If no discrepancy, mention 'No discrepancy'."
                                        },
                                        "resolution_suggestion": {
                                            "type": "string",
                                            "description": "Suggestion on how to handle discrepancies or clean the data. If no discrepancy, mention 'No action required'."
                                        },
                                        "action_needed": {
                                            "type": "string",
                                            "description": "'Yes' if the current data type differs from the inferred data type or there are missing/invalid values, otherwise 'No'."
                                        }
                                    },
                                    "required": [
                                        "column_name",
                                        "inferred_data_type",
                                        "reason_for_discrepancy",
                                        "resolution_suggestion",
                                        "action_needed"
                                    ]
                                }
                            }
                        },
                        "required": ["column_analysis"]
                    }
                }
            ],
            "function_call": {
                "name": "validate_column_data"
            }
        }

        return query_llm(json_data)
    except Exception as e:
        print("Error in validate_column function:", e)
        raise



def basic_analysis(df,folder_path):
    numeric_summary = df.describe()
    missing=df.isnull().sum()
    numeric_df=df.select_dtypes(include=['number'])
    cat_column = df.select_dtypes(include=['object']).columns
    if len(numeric_df.columns)>5:
        numericdf = numeric_df.iloc[:,:5]
    elif 1 < len(numeric_df.columns) <= 5:
        numericdf = numeric_df
    try:
        plt.figure(figsize=(15,15))
        sns.heatmap(numericdf.corr() ,cmap="coolwarm",annot=True)
        plt.title("Correlation Analysis")
        img_path1 = os.path.join(folder_path,"heatmap.png")
        plt.savefig(img_path1)
        plt.clf()
    except Exception as e:
        print("Error in basic_analysis function:", e)

    img = []
    if img_path1:
        img.append({'analysis_name': "Numerical heatmap analysis","description" : "correlation analysis for numerical columns using heatmap", "image_path" : img_path1})

    return numeric_df.describe(),img


def importent_column(df,file_name):
    """
    Identifies important and potentially irrelevant columns in the dataset based on sample data.

    Parameters:
        df (pd.DataFrame): The DataFrame to analyze.

    Returns:
        dict: Analysis results identifying important and irrelevant columns.
    """
    try:
        data = {}
        for column in df.columns:
            data[column] = df[column].head(10).astype(str).tolist()

        sample_data = [{"column_name": column, "sample_data": data[column]} for column in df.columns]

        json_data = {
            "model": "gpt-4o-mini",
            "messages": [
                {
                    "role": "system",
                    "content": "You are a dataset analysis assistant. Provide structured and concise output."
                },
                {
                    "role": "user",
                    "content": f'''Analyze the following dataset sample form {file_name}:

                    1. What is the possible description or context of this dataset?  Explain your reasoning.
                    2. give this dataset a suitable 3-5 word name?
                    3. Identify the columns that seem most important for analysis and explain why they are significant.
                    4. If there are columns that appear irrelevant or redundant, list them.

                    Strictly base your analysis on the given data and column names without making assumptions beyond what the data suggests.
                    Here is the dataset sample:

                    {sample_data}
                    '''
                }
            ],
            "functions": [
                {
                    "name": "analyze_dataset",
                    "description": "Analyze the dataset to infer its purpose, identify important columns, and suggest how to focus further analysis.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "dataset_description": {
                                "type": "string",
                                "description": "The inferred description or context of the dataset based on column names and sample data."
                            },
                            "dataset_name": {
                                "type": "string",
                                "description": "based on the description give the dataset a 3-5 word name."
                            },
                            "important_columns": {
                                "type": "array",
                                "description": "List of columns deemed most important for analysis, along with reasons for their importance.",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "column_name": {
                                            "type": "string",
                                            "description": "The name of the important column."
                                        },
                                        "reason": {
                                            "type": "string",
                                            "description": "The reason why this column is important for analysis."
                                        }
                                    },
                                    "required": ["column_name", "reason"]
                                }
                            },
                            "irrelevant_columns": {
                                "type": "array",
                                "description": "List of columns that appear irrelevant or redundant for analysis.",
                                "items": {
                                    "type": "string"
                                }
                            }
                        },
                        "required": ["dataset_descripition","dataset_name" , "important_columns"]
                    }
                }
            ],
            "function_call": {"name": "analyze_dataset"}
        }

        return query_llm(json_data)
    except Exception as e:
        print("Error in importent_column function:", e)
        raise

def genrate_code(action):
    """
    Generates Python code to resolve discrepancies in the dataset and ensure data consistency.

    Parameters:
        action (dict): Details of columns and discrepancies that require resolution.

    Returns:
        dict: Generated Python code and explanations for corrections.
    """
    try:
        json_data = {
            "model": "gpt-4o-mini",
            "messages": [
                {
                    "role": "system",
                    "content": "You are a dataset analysis assistant. Provide structured and concise output using the 'generate_correction_code' function."
                },
                {
                    "role": "user",
                    "content": f'''
                        You are a dataset analysis assistant. Based on the provided column details and their sample data, generate Python code to resolve discrepancies and ensure data consistency.
                        For each column:
                        1. Analyze the sample data and understand the inferred correct data type.
                        2. Provide Python code to address any discrepancy and convert the data to the inferred correct type.
                        4. Consider the reason for the discrepancy and the suggested resolution while providing the correction.
                        The code should be optimized and consider all the corner cases. It should not take much time to run.
                        Assume data is already loaded in a variable named "df".
                        The provided code should not have comments. Use only pandas, sklearn, and numpy libraries only. Write the import statement at the beginning of the code.
                        Here are the columns to analyze:
                        {action}
                    '''
                }
            ],
            "functions": [
                {
                    "name": "generate_correction_code",
                    "description": "Provide Python code to resolve discrepancies in column data types based on inferred data type and resolution suggestion.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "column_corrections": {
                                "type": "array",
                                "description": "Code corrections for each column requiring action.",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "column_name": {
                                            "type": "string",
                                            "description": "The name of the column being corrected."
                                        },
                                        "correction_code": {
                                            "type": "string",
                                            "description": "Python code snippet to apply the correction."
                                        },
                                        "explanation": {
                                            "type": "string",
                                            "description": "Explanation of the logic behind the correction code."
                                        }
                                    },
                                    "required": ["column_name", "correction_code", "explanation"]
                                }
                            }
                        },
                        "required": ["column_corrections"]
                    }
                }
            ],
            "function_call": {
                "name": "generate_correction_code"}
        }

        return query_llm(json_data)
    except Exception as e:
        print("Error in genrate_code function:", e)
        raise

def purpose_analysis(result):
    """
    Generates proposed analyses for a dataset based on its description and columns.

    Args:
        result (dict): A dictionary containing dataset details.

    Returns:
        dict: A structured response containing proposed analyses in JSON format.
    """
    try:
        json_data = {
            "model": "gpt-4o-mini",
            "messages": [
                {
                    "role": "system",
                    "content": "You are a data analysis assistant. Based on dataset descriptions and column details, provide structured analysis tasks that can be performed on the dataset."
                },
                {
                    "role": "user",
                    "content": f'''Based on the provided dataset details, propose four types of analyses we can perform. It can be statistical or any type of groundbreaking analysis.
                    Include analyses that leverage important columns while addressing the dataset's purpose.
                    Guidelines:
                    1.Avoid Overcrowded Visualizations: Skip visualizations that involve too many unique categorical values, which would clutter charts (e.g., histograms or pie charts for high-cardinality columns).
                    2.Top 'N' Strategy: For columns with many unique categorical values (more than 10), such as names, categories, or other identifiers, focus on the 'Top N' entries based on a relevant criterion (e.g., frequency, total value, or average measure).
                    3.Continuous Numeric Columns: For continuous numeric columns, suggest appropriate analyses like distributions (histograms, KDE plots) or box plots. If further analysis is needed, propose binning or grouping the values for better insights.
                    4.Purpose-Driven Analyses: Ensure each proposed analysis has a clear purpose, provides actionable insights, and aligns with the dataset’s goals.
                    Additional Notes:
                    - Ensure the analyses are diverse, covering statistical, visual, and logical insights.
                    - Avoid repetitive suggestions. Each analysis should address a unique aspect of the dataset.
                    - Adapt your suggestions to the dataset description, keeping in mind the type and range of columns.
                    Here is the dataset details:
                    {result}
                    '''
                }
            ],
            "functions": [
                {
                    "name": "propose_analyses",
                    "description": "Propose potential analysis tasks based on dataset details, including names, descriptions, and relevant columns.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "analyses": {
                                "type": "array",
                                "description": "List of analyses proposed based on the dataset.",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "analysis_name": {
                                            "type": "string",
                                            "description": "Name of the analysis task."
                                        },
                                        "description": {
                                            "type": "string",
                                            "description": "Description of the analysis task, including its objective and relevance to the dataset's purpose."
                                        },
                                        "key_columns": {
                                            "type": "array",
                                            "description": "List of important columns involved in this analysis.",
                                            "items": {
                                                "type": "string"
                                            }
                                        }
                                    },
                                    "required": ["analysis_name", "description", "key_columns"]
                                }
                            }
                        },
                        "required": ["analyses"]
                    }
                }
            ],
            "function_call": {
                "name": "propose_analyses"
            }
        }
        return query_llm(json_data)
    except Exception as e:
        raise RuntimeError(f"Error in purpose_analysis: {str(e)}")

def analysis_code(analysis,folder_name):
    
    """
    Generates Python code for performing specified analyses on a dataset.

    Args:
        analysis (dict): A dictionary describing the analyses to perform.

    Returns:
        dict: A structured response containing Python code snippets for each analysis.
    """
    try:
        json_data = {
            "model": "gpt-4o-mini",
            "messages": [
                {
                    "role": "system",
                    "content": "You are a Python data analysis assistant. Generate code for data analysis tasks based on the provided analysis descriptions. Use matplotlib and seaborn for visualizations, scipy.stats for statistical analysis, and sklearn for machine learning algorithms."
                },
                {
                    "role": "user",
                    "content": f'''Generate Python code for the following analyses. Ensure the use of matplotlib and seaborn for visualizations, scipy.stats for any statistical models, and sklearn for machine learning algorithms if applicable. 
                    while generating code keep mind that you don't write extra excape charactor.
                    Each code snippet should include relevant imports, data handling steps, and visualizations where appropriate.
                    Assume that the dataset is already loaded and stored in "df". Save the resulting images in folder "{folder_name}"  as .png format.(don't show the plot in terminal just save the image in the folder and do plt.close())
                    Ensure the axis labels, titles, and tick labels are **fully visible** by using `plt.tight_layout()` or adjusting figure sizes appropriately.
                    Use gradient or pastel colors for histograms, plots, pie charts, and similar visualizations.
                    The analyses are:
                    {analysis}
                    '''
                }
            ],
            "functions": [
                {
                    "name": "generate_analysis_code",
                    "description": "Generate Python code for data analyses based on descriptions, using matplotlib, seaborn, scipy.stats, and sklearn.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "analysis_code": {
                                "type": "array",
                                "description": "Python code snippets for each analysis task.",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "analysis_name": {
                                            "type": "string",
                                            "description": "The name of the analysis."
                                        },
                                        "code_snippet": {
                                            "type": "string",
                                            "description": "Python code for performing the analysis. The code should not include comments."
                                        },
                                        "explanation": {
                                            "type": "string",
                                            "description": "Explanation of the code and its purpose in addressing the analysis."
                                        },
                                        "image_path": {
                                            "type": "string",
                                            "description": "Path to the generated image."
                                        }
                                    },
                                    "required": ["analysis_name", "code_snippet", "explanation","image_path"]
                                }
                            }
                        },
                        "required": ["analysis_code"]
                    }
                }
            ],
            "function_call": {
                "name": "generate_analysis_code"
            }
        }
        return query_llm(json_data)
    except Exception as e:
        raise RuntimeError(f"Error in analysis_code: {str(e)}")

def analyze_image(analysis):
    """
    Analyze an image using LLM to generate structured insights.

    Parameters:
    - image_path (str): The file path of the image to be analyzed.
    - prompt_description (str): A description of the type of analysis needed for the image.

    Returns:
    - dict: Structured analysis response from the LLM.
    """
    try:
        # Encode the image in base64
        with open(analysis["image_path"], "rb") as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode("utf-8")

        # LLM JSON payload
        json_data = {
            "model": "gpt-4o-mini",
            "messages": [
                {
                    "role": "system",
                    "content": "You are an advanced image analysis assistant. Provide detailed and structured insights based on images provided."
                },
                {
                    "role": "user",
                    "content":  [
                        {
                            "type": "text",

                            "text": f'''see the image and give the intresting finding from the chart.
                            the language should be engaging.
                            here is the overview of the analysis
                              analysis name : {analysis["analysis_name"]}
                              description : {analysis["description"]}
                             '''
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "detail": "low",
                                "url": f"data:image/png;base64,{encoded_image}"
                            }
                        }
                    ]
                }
            ],
            "functions": [
                {
                    "name": "analyze_image_content",
                    "description": "Provide analysis of image in a engaging and intresting way",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "analysis_name":{
                                "type": "string",
                                "description": "The name of the analysis."
                            },
                            "insights": {
                                "type": "string",
                                "description": "insights or key findings from the image."
                            }
                        },
                        "required": ["analysis_name", "insights"]
                    }
                }
            ],
            "function_call": {
                "name": "analyze_image_content"
            }
        }

        return query_llm(json_data)
    except Exception as e:
        raise RuntimeError(f"Error in analyze_image: {str(e)}")
    
def markdown_genratar(dataset_name,columns, dataset_description,describe,dataset_size,basic_result,final_result):
    json_data = {
            "model": "gpt-4o-mini",
            "messages": [
                {
                    "role": "system",
                    "content": "You are a dataset analysis assistant. Provide structured and concise markdown content using the the dataset summay and analysis."
                },
                {
                    "role": "user",
                    "content": f'''
                        You are a dataset analysis assistant. Based on the provided data give a summary in markdown file content.
                        belwo is the structure of the markdown file structure.
                        1. first the title of the file which is {dataset_name}
                        2. after this descripition of the dataset which is {dataset_description}
                        3. a heading called Dataset Summary
                           dataset size : {dataset_size}
                           columns names :{columns}

                           - a subheading called Numerical summary statistics
                              it contains the statistical summary of numerical colums in a markdown table.
                              {describe}
                           - EDA : this will have {basic_result}

                       4. a heading called key analysis:
                          use key findings:{final_result}

                          if key findings exists then the make pointer for each finding
                            1. analysis name
                                image ![Alt Text](image_name) (here the image name should be as it is don't put "/" at the start)
                                - finding from this analysis
                      5. then summaries the whole report. in 2-3 lines


                      so this is the structure of the markdown report. use engaging style and strictly follow the markdown structure.



                    '''
                }
            ]
    }
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {AIPROXY_TOKEN}"}
    url = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"
    response = requests.post(url=url, headers=headers, json=json_data)
    return response.json()["choices"][0]["message"]["content"]




def main():
    try:
        if len(sys.argv) != 2:
            print("Usage: uv run autolysis.py <input_filename.csv>")
            sys.exit(1)

        file_name = sys.argv[1]
        if not os.path.exists(file_name):
            print("File not found in the current working directory.")
            sys.exit(1)
        

        # Create a folder in the root directory with the starting name of the file
        
        folder_name = os.path.splitext(file_name)[0]
        folder_path = os.path.join(os.getcwd(), folder_name)
        os.makedirs(folder_path, exist_ok=True)
        
        with open(file_name, 'rb') as file:
            raw_data = file.read()
            result = chardet.detect(raw_data)
            encoding = result['encoding']
        # Load the dataset
        df = pd.read_csv(file_name,encoding=encoding)
    except Exception as e:
        print("Error during initial setup or file loading:", str(e))
        return "Process terminated due to initial setup failure."

    try:
        # Identify important columns
        result = json.loads(importent_column(df,file_name))
        dataset_name = result["dataset_name"]
        dataset_description = result["dataset_description"]
        important_columns = [col["column_name"] for col in result.get("important_columns", [])]

    except Exception as e:
        print("Error in identifying important columns:", str(e))
        important_columns = []

    try:
        # Validate columns
        validation_result = json.loads(validate_column(df, important_columns))
        actions = []

        for column in validation_result.get("column_analysis", []):
            if column.get("action_needed") == "Yes":
                actions.append({
                    "column_name": column["column_name"],
                    "sample_data": df[column["column_name"]].head(10).tolist(),
                    "inferred_data_type": column["inferred_data_type"],
                    "reason_for_discrepancy": column["reason_for_discrepancy"],
                    "resolution_suggestion": column["resolution_suggestion"]
                })

        if actions:
            actions.append({"dataset_purpose": result.get("dataset_purpose")})
            actions.append({"dataset_size": (df.shape[0], len(important_columns))})

            # Generate and execute correction code
            try:
                correction_code_result = json.loads(genrate_code(actions))
                for correction in correction_code_result.get("column_corrections", []):
                    try:
                        exec(correction["correction_code"])
                    except Exception as exec_error:
                        print(f"Error executing correction code for column {correction['column_name']}: {str(exec_error)}")
            except Exception as e:
                print("Error in generating or executing correction code:", str(e))
    except Exception as e:
        print("Error in column validation:", str(e))

    try:
        # Analyze dataset purpose and propose analyses
        rd = {"dataset_purpose": result.get("dataset_purpose"),"columns" : []}
        for i in result["important_columns"]:
            if i["column_name"] in df.select_dtypes(include=['object']).columns:
                rd["columns"].append({"column_name":i["column_name"],"data_type":"string","unique_values":len(df[i["column_name"]].unique()),"sample_data" : df[i["column_name"]].head(10).tolist()})
            elif i["column_name"] in df.select_dtypes(include=["number"]).columns:
                rd["columns"].append({"column_name":i["column_name"],"data_type":"numeric","summary_statistics" : df.describe()[i["column_name"]] ,"sample_data" : df[i["column_name"]].head(10).tolist()})
            else :
                rd["columns"].append({"column_name":i["column_name"],"data_type":"other","sample_data" : df[i["column_name"]].head(10).tolist()})

        purpose_analysis_result = json.loads(purpose_analysis(rd))
        analyses = purpose_analysis_result.get("analyses", [])

        # Add sample data for analyses
        data_samples = {column: df[column].head(10).tolist() for column in important_columns}
        sample_data = [{"column_name": column, "sample_data": data_samples[column]} for column in important_columns]
        analyses.append(sample_data)

        # Generate and execute analysis code
        try:
            analysis_code_result = json.loads(analysis_code(analyses,folder_name))
            final_analysis = []
            for i, analysis in enumerate(analysis_code_result.get("analysis_code", [])):
                try:
                    exec(analysis["code_snippet"])
                    for i in analyses:
                        if analysis["analysis_name"] == i["analysis_name"]:
                            final_analysis.append({"analysis_name":i["analysis_name"] ,"description":i["description"] , "image_path" : analysis["image_path"]  })
                except Exception as exec_error:
                    print(f"Error executing analysis code for {analysis['analysis_name']}: {str(exec_error)}")
        except Exception as e:
            print("Error in generating or executing analysis code:", str(e))
    except Exception as e:
        print("Error in dataset purpose analysis or proposing analyses:", str(e))

    try:
        basic_result=[]
        final_result=[]
        if len(final_analysis) == 0 :
            describe , img = basic_analysis(df,folder_path)
            for i in img:
                basic_result.append(json.loads(analyze_image(i)))
        elif len(final_analysis) > 0:
            new_folder_path = os.path.join(folder_path, "extra")
            os.makedirs(new_folder_path, exist_ok=True)
            describe, img = basic_analysis(df,new_folder_path)
            for i in img:
                basic_result.append(json.loads(analyze_image(i)))
            shutil.rmtree(new_folder_path)
            for i in final_analysis:
                analysis_result = json.loads(analyze_image(i))
                analysis_result["image_name"] = i["image_path"].split("/")[1]
                final_result.append(analysis_result)
            
    except Exception as e:
        print("Error in image analysis:", str(e))

    dataset_size = df.shape
    markdown_content = markdown_genratar(dataset_name,df.columns, dataset_description,describe,dataset_size,basic_result,final_result)
    open(os.path.join(folder_path, "README.md"), "w").write(markdown_content)
    return "Analysis completed with potential errors logged."


if __name__ == "__main__":
    result_message = main()
    print(result_message)

