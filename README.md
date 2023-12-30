# yt-comment-sentiment-analyzer
YT Comment Analyzer using LSTM


# YouTube Comment Sentiment Analyzer
The YouTube Comment Analyzer is a web application that allows users to analyze the sentiment of comments on a YouTube video. By simply providing the YouTube video URL, the application leverages the YouTube API and a pretrained LSTM model to fetch all the comments and categorize them into positive, negative, or neutral sentiments. This tool serves as a powerful way to understand the overall reception of a video based on the sentiments expressed by its viewers.

## Features
- #### User-friendly Interface: 
    The web application provides an intuitive interface where users can easily input the YouTube video URL and obtain the sentiment analysis results.

- #### YouTube API Integration: 
    The backend is integrated with the YouTube API, which facilitates the extraction of comments from the specified video, enabling seamless data retrieval.

- #### Sentiment Analysis using LSTM Model: 
    Utilizing a pretrained LSTM model, which is trained on IMDb dataset and gained an accuracy of 86.67%, the application accurately assesses the sentiment of each comment, categorizing them as positive, negative, or neutral based on their content.

- #### Real-time Analysis: 
    The application performs sentiment analysis in real-time, providing users with immediate feedback on the sentiments expressed in the comments.

- #### Django Web Framework: 
    The backend of the application is developed using Django, ensuring robustness, scalability, and maintainability.


## How to Use
- #### Download the Project: 
    To get started, first, download the project from the GitHub repository. You can do this by clicking on the "Code" button and selecting "Download ZIP." Alternatively, you can clone the repository using the following command in your terminal:

    `git clone https://github.com/amulyakpadhan/yt-comment-sentiment-analyzer.git`

- #### Install Dependencies: 
    After downloading the project, navigate to the project directory using the terminal or command prompt. Then, install all the required libraries and dependencies by running the following command:

    `pip install -r requirements.txt`

- #### YouTube API Key: 
    The YouTube Comment Analyzer requires a valid YouTube API key to fetch comments from the YouTube video. If you don't have one, you can obtain it by following the instructions on the YouTube API documentation.

- #### Configure API Key: 
    Once you have the API key, open the project in your favorite code editor and locate the views.py file of the myapp directory within the project's Django app directory. Replace YOUR_KEY in the DEVELOPER_KEY = "YOUR_KEY" with your actual YouTube API key.

- #### Run the Server: 
    Now that you have installed all the dependencies and configured the API key, you can run the Django development server using the following command:

    `python manage.py runserver`

- #### Access the Web Application: 
    With the development server running, open your web browser and go to http://localhost:8000/ or http://127.0.0.1:8000/. You will be greeted with the YouTube Comment Analyzer web page.

- #### Analyze Comments: 
    Enter the URL of the YouTube video you wish to analyze in the provided input field and click on the "Analyze" button. The application will fetch the comments and perform sentiment analysis, displaying the results on the page.

## Deployment
The YouTube Comment Analyzer is built on the Django web framework, allowing for easy deployment to various hosting platforms. You can deploy the application to a web server of your choice using WSGI servers like Gunicorn or by using platforms like Heroku.

By combining the power of the YouTube API, LSTM Machine Learning model, and Django framework, the YouTube Comment Analyzer offers a straightforward yet effective way to gain valuable insights into the sentiments expressed by viewers of any YouTube video. Whether it's for content creators seeking audience feedback or researchers studying online reactions, this tool provides a comprehensive sentiment analysis solution in a user-friendly web format. Try it out and explore the sentiments behind the comments on your favorite YouTube videos!
