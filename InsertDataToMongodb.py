def insertFeedbackValuetoDb(botResponseInJson):

    d=userDetails(botResponseInJson)
    
    existingRecord = db.feedbackData.find_one({'$and':[{'bot_id':(botResponseInJson['botID'])},
                                        {'user_details': d }]})
    
    botRecord = {
                    "bot_id" : botResponseInJson['botID'],
                    "user_details": d,

                    "chat_history": [{
                        "user_utterance" : botResponseInJson['message'],
                        "bot_response": botResponseInJson['bot_response'],
                        "createdAt" : f"{datetime.datetime.now():%A, %B %d, %Y %I:%M %p}"

                    }]
                }
    if existingRecord is None:
        record = db.feedbackData.insert(botRecord)
    else:
        oldvalues = existingRecord                      
                      
        newvalues = {
                '$push': {
                    "chat_history": {
                                    "user_utterance" : botResponseInJson['message'],
                                    "bot_response": botResponseInJson['bot_response'],
                                    "createdAt" : f"{datetime.datetime.now():%A, %B %d, %Y %I:%M %p}"

                                    }
                        }
                    }
        record = db.feedbackData.update(oldvalues,newvalues)
