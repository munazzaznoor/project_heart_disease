from django.shortcuts import render
import pickle

def home(request):
    return render(request,"blog/base.html",{})

def getPredictions(male,age,education,currentsmoke,cigsPerDay,BPMeds,prevalentStroke,prevalentHyp,diabetes,totChol,sysBP,diaBP,BMI,heartRate,glucose):

    model=pickle.load(open("accounts/ML/model and data/model_pickle.sav","rb"))
    scaled=pickle.load(open("accounts/ML/model and data/scaler.sav","rb"))
    prediction=model.predict(scaled.transform([[male,age,education,currentsmoke,cigsPerDay,BPMeds,prevalentStroke,prevalentHyp,diabetes,totChol,sysBP,diaBP,BMI,heartRate,glucose]]))

    if prediction==0:
        return 'no'
    elif prediction==1:
        return 'yes'
    else:
        return 'error'


def result(request):
    male=int(request.GET["male"])
    age=int(request.GET["age"])
    education=int(request.GET["education"])
    currentSmoker=int(request.GET["currentSmoker"])
    cigsPerDay=int(request.GET["cigsPerDay"])
    BPMeds=int(request.GET["BPMeds"])
    prevalentStroke=int(request.GET["prevalentStroke"])
    prevalentHyp=int(request.GET["prevalentHyp"])
    diabetes=int(request.GET["diabetes"])
    totChol=int(request.GET["totChol"])
    sysBP=int(request.GET["sysBP"])
    diaBP=int(request.GET["diaBP"])
    BMI=int(request.GET["BMI"])
    heartRate=int(request.GET["heartRate"])
    glucose=int(request.GET["glucose"])

    result=getPredictions(male,age,education,currentSmoker,cigsPerDay,BPMeds,prevalentStroke,prevalentHyp,diabetes,totChol,sysBP,diaBP,BMI,heartRate,glucose)
    return render(request, 'blog/result.html', {"result":result})