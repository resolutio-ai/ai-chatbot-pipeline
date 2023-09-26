# ai-chatbot-pipeline
Backend for AI Chatbot with a serving wrapper in Flask

### Getting Started

In order to run the pipeline, please follow the below steps.

1. Download and Install Python 3.11.x (this version uses 3.11.2)
2. Clone this repository
3. Create a python virtual environment like below
```commandline
python3 -m venv /path/to/env

example: `python3 -m venv ai_env`
```
4. Activate the virtual environment like below.
```commandline
source activate /path/to/env/bin/activate

example: `source activate ai_env/bin/activate`
```
5. Install the required Python packages as followed
```commandline
cd ai-chatbot-pipeline/
pip install -r requirements.txt
```
6. Next, place the `.env` file (sent separately for security reasons) outside the repository directory.
7. Now go to `/ai-chatbot-pipeline` directory and from terminal, run app.py like below.
```commandline
python app.py
```
7. Once the app is launched, from another terminal (or terminal tab), run the bash script to make cURL requests.
```commandline
./run.sh
```
8. In the first terminal, you should see the main menu like below. Use Arrow to select any option and press Enter.
```commandline
 What do you need help with today?: General IP Queries
 > General IP Queries
   Azuki License
   Viral Public License
   BYAC License
   MAYC License

```
9. Once an option is selected, you will be asked to give a prompt. 
10. Write your question as a prompt and hit enter. First time it may take a while to get back a response.
11. The response will be printed in the terminal for your reference. Additionally, the data (answer + chat_history) will be written in `chats.json` file in real-time.

