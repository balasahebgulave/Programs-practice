def getQuiz(botResponseInJson):
    quizdata=db.Quiz.find_one({"bot_id" : ObjectId(botResponseInJson['botID'])})
    questions=quizdata['questions']
    qList = random.sample([i for i in range(len(questions))], 1)
    qList = [questions[i] for i in qList]
    botResponseInJson['qList']=qList


def finalQuizResponse(botResponseInJson):

    quizdata=db.Quiz.find_one({"bot_id" : ObjectId(botResponseInJson['botID'])})
    questions=quizdata['questions']
    Answers=[i["rightanswer"] for i in questions]


    if  len(botResponseInJson['quick_reply'])==1:
        
        scoreCounter=int(botResponseInJson['score'])

        if botResponseInJson['message'] in Answers:
            scoreCounter+=1
            botResponseInJson['score']=scoreCounter
            

        if botResponseInJson['quick_reply'][0]['quick_reply_intent']=="":
            botResponseInJson['quick_reply'] = []
            for score in str(botResponseInJson['score']):
                botResponseInJson['bot_response']=('Your Score Is : '+ score)
                botResponseInJson['score']=0
                break

        else:
            botResponseInJson['bot_response']=botResponseInJson['qList'][0]['question']
            reply=[]
            q=botResponseInJson['qList'][0]  
            for option in q['options']:
                data={"quick_reply_type": "text", "text": "", "quick_reply_intent": ""}
                data["text"]=q['options'][option]
                data["quick_reply_intent"]=botResponseInJson['quick_reply'][0]['quick_reply_intent']
                reply.append(data)
            botResponseInJson['quick_reply']=reply
