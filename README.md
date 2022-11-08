# GBX: Gradient Boosting eXplorer for Credit Risk Modeling 

GBX is

* a solution based on gradient boosting to credit risk modeling

GBX supports

* Ad hoc analysis on credit assessment data





## Installation

```bash
git clone https://github.com/imyizhang/gbx.git 
```



### Install Dependencies

Under GBX working directory

```bash
pip install -r requirements.txt
```





## Quickstart

Under GBX working directory

```bash
streamlit run app.py
```



### GBX on Amazon EC2

#### Setup Amazon EC2 Instance



#### Connect to Amazon EC2 Instance via SSH

```bash
ssh -i /path/key-pair-name.pem instance-user-name@instance-public-dns-name
```

For instance, `ssh -i chengdu80-EMlyon.pem ubuntu@3.74.252.194`.



#### Setup Environment and Install GBX



#### Run GBX in the Background on Amazon EC2 Instance

* Start a new session for command `streamlit`

  ```bash
  tmux new -s Streamlit
  ```



* Start Streamlit application server

  ```bash
  streamlit run gbx/app.py
  ```

  For instance, our Streamlit application is running at http://3.74.252.194:8501/.



* Detach from session

  `Ctrl` + `b` `d`



##### Run Jupyter Notebook in the Background on Amazon EC2 Instance

* Start a new session for command `jupyter notebook`

  ```bash
  tmux new -s Jupyter
  ```



* Start Jupyter Notebook server

  ```bash
  jupyter notebook --ip='*' --no-browser
  ```

  For instance, our Jupyter Notebook is running at http://3.74.252.194:8888/.



* Detach from session

  `Ctrl` + `b` `d`



##### tmux Cheat Sheet

* Start a new session with name

  ```bash
  tmux new-session -s session-name
  ```

  or

  ```bash
  tmux new -s session-name
  ```

  

* Detach from a session

  `Ctrl` + `b` `d`

  

* Attach to a session with name

  ```bash
  tmux attach-session -t session-name
  ```

  or

  ```bash
  tmux attach -t session-name
  ```

  

* List all sessions

  ```bash
  tmux list-sessions
  ```

  or

  ```bash
  tmux ls
  ```

  

* Kill a session with name

  ```bash
  tmux kill-session -t session-name
  ```



##### lsof Cheat Sheet

* List the listening ports and processes

  ```bash
  lsof -i -P -n
  ```

  

##### kill Cheat Sheet

* Kill a process by PID

  ```bash
  kill -9 PID
  ```






## Contribution





## License

GBX has an BSD-3-Clause license, as found in the [LICENSE](https://github.com/imyizhang/gbx/blob/main/LICENSE) file.



