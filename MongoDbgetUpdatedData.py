

#program for taking data from mongodb collection from that pick only updated data.


transaction = db.feedbackData.find_one( {'$and':[{'bot_id':(botResponseInJson['botID'])},
                                        {'user_details': d }]},{"chat_history":1} )
                                        
chat=[]

for history in transaction["chat_history"]:
  chat.append(history)
  
updated_transaction=(list({d['bot_response']:d for d in chat}.values()))

print('updated_transaction : ', updated_transaction)



