from taipy.gui import Gui, State, notify
from function import *

context = "以下是與AI助理的對話。 助理樂於助人、有創意、聰明且非常友善。 " \
          " \n\n人類：你好，你是誰？ 今天我能為您提供什麼幫助？ "

conversation = {
  "Conversation": [
    "你是誰?", "我是辦公室檔案小助手, 可以回答檔案內容"
  ]
}
current_user_message = ""
past_conversations = []
selected_conv = None
selected_row = [1]
content = ""
chunk_size = 150
chunk_overlap = 0
chain = None
skiprows = None
separators = "'\\n\\n', '\\n', ' ', ''"
is_csv = False

def on_init(state):
  state.context = context
  state.conversation = conversation
  state.current_user_message = current_user_message
  state.past_conversations = past_conversations
  state.selected_conv = selected_conv
  state.selected_row = selected_row
  state.content = content
  state.chunk_size = chunk_size
  state.chunk_overlap = chunk_overlap
  state.chain = chain
  state.separators = separators
  state.skiprows = skiprows
  state.is_csv = is_csv


def RAG(state):
  print(state.content)
  print(state.separators)
  notify(state, "info", "載入中...")
  if 'pdf' in state.content:
    print('loading pdf...')
    docs = pdf_load(state.content)
    print('pdf loaded.')
  else:
    docs = office_file(state.content)
  notify(state, "info", "分割段落...")
  splits = splitter(docs, eval(f"[{state.separators}]"),
                    int(state.chunk_size),
                    int(state.chunk_overlap))
  notify(state, "info", "轉向量...")
  state.chain = rag(splits)
  notify(state, "info", "完成!")
  print('完成')


def csv_file(state):
  if 'csv' in state.content and state.skiprows is not 'None':
    state.is_csv = True
    state.chain = pandas_agent(state.content, int(state.skiprows))
  elif not 'csv' in state.content:
    state.is_csv = False
  else:
    notify(state, "error", "請先輸入 CSV 檔案參數")


def request(state, prompt):
  if state.chain:
    response = state.chain.invoke(prompt)
    if state.is_csv == True:
      return response['output']
    else:
      return response.replace("\n", "")
  else:
    notify(state, "error", "請先上傳檔案")
    return "請先上傳檔案"


def update_context(state) -> str:
  state.context += f"Human: \n {state.current_user_message}\n\n AI:"
  answer = request(state, state.context)
  state.context += answer
  state.selected_row = [len(state.conversation["Conversation"]) + 1]
  return answer


def send_message(state: State) -> None:
  notify(state, "info", "傳送中...")
  answer = update_context(state)
  conv = state.conversation._dict.copy()
  conv["Conversation"] += [state.current_user_message, answer]
  state.current_user_message = ""
  state.conversation = conv
  notify(state, "success", "收到回覆!")


def style_conv(state: State, idx: int, row: int) -> str:
  if idx is None:
    return None
  elif idx % 2 == 0:
    return "user_message"
  else:
    return "gpt_message"


def on_exception(state, function_name: str, ex: Exception) -> None:
  notify(state, "error", f"An error occured in {function_name}: {ex}")


def reset_chat(state: State) -> None:
  state.past_conversations = state.past_conversations + [[
      len(state.past_conversations), state.conversation
  ]]
  state.conversation = {
    "Conversation": [
      "你是誰?", 
      "我是 Youtube 小助手, 可以回答影片內容"
    ]
  }


def tree_adapter(item: list):
  identifier = item[0]
  if len(item[1]["Conversation"]) > 3:
    return (identifier, item[1]["Conversation"][2][:50] + "...")
  return (item[0], "Empty conversation")


def select_conv(state: State, var_name: str, value) -> None:
  print(value)
  state.conversation = state.past_conversations[value[0][0]][1]
  state.context = "以下是與AI助理的對話。 助理樂於助人、有創意、聰明且非常友善。"\
                  " \n\n人類：你好，你是誰？ 今天我能為您提供什麼幫助？ "
  for i in range(2, len(state.conversation["Conversation"]), 2):
    state.context += f"Human: \n {state.conversation['Conversation'][i]}\n\n AI:"
    state.context += state.conversation["Conversation"][i + 1]
  state.selected_row = [len(state.conversation["Conversation"]) + 1]


past_prompts = []

page = """
<|toggle|theme|>

<|layout|columns=300px 1|

<|part|class_name=sidebar|
# Youtube AI **Chat**{: .color-primary} # {: .logo-text}
<|新對話|button|class_name=fullwidth plain|id=reset_app_button|on_action=reset_chat|>

### 過去的對話記錄 ### {: .h5 .mt2 .mb-half}
<|{selected_conv}|tree|lov={past_conversations}|class_name=past_prompts_list|multiple|adapter=tree_adapter|on_change=select_conv|>
|>
<|part|class_name=sidebar scrollable|
### 請選擇檔案 ###
<|{content}|file_selector|extensions=.docx,.pptx,.csv,.xlsx,.pdf|on_action=csv_file|drop_message=拖至此處上傳|>
<|{content}|input|label=檔案名稱|>
<|part|>
<|{separators}|input|label=分割字串|>
<|{chunk_size}|input|label=分割字元數|>
<|{chunk_overlap}|input|label=重複字元數|>
<|RAG 處理|button|class_name=small-button plain|id=open_button|on_action=RAG|>

### CSV 檔案參數 ###
<|{skiprows}|input|label=忽略行數|>

<br/>
<|part|class_name=p2 align-item-bottom table|
<|{conversation}|table|style=style_conv|show_all|selected={selected_row}|rebuild|>

<|part|class_name=card mt1|
<|{current_user_message}|input|label=你的訊息...|on_action=send_message|class_name=fullwidth|change_delay=-1|>
|>
|>
|>
|>
"""

if __name__ == "__main__":
  gui = Gui(page=page)
  gui.run(margin="0em", port=8080, host='0.0.0.0', title="辦公室檔案問答機器人")
