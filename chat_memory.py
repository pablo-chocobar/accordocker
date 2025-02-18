class ChatMemory:
    def __init__(self):
        self.conversations = []
        
    def add_conversation(self, user_msg, system_msg):
        self.conversations.append({
            'user': user_msg,
            'system': system_msg
        })
        
    def get_history(self):
        return self.conversations
        
    def clear_history(self):
        self.conversations = []