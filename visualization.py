import os
import re
import sys
import pickle
import traceback
import eli5
from eli5.lime import TextExplainer
from eli5.formatters import format_as_html
from sklearn.pipeline import make_pipeline
from lime.lime_text import LimeTextExplainer

###########################################################################################################################
# Author        : Tapas Mohanty  
# Modified      : 
# Reviewer      :
# Functionality : Import Model
###########################################################################################################################
class MyCustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "__main__":
            module = "custom_token"
        return super().find_class(module, name)
        
def loadmodel(pRootDir, pModelName, pIntent):
    model_loc = pRootDir + '\\' +  str(pModelName) + '\\' + str(pModelName[6:]) +  '_Model' + '\\' + str(pIntent) + ".model.pkl"
    with open(model_loc, 'rb') as f:
        unpickler = MyCustomUnpickler(f)
        model = unpickler.load()
    return (0, model)

###########################################################################################################################
# Author        : Tapas Mohanty  
# Modified      : 
# Reviewer      :
# Functionality : Explanation of prediction through Lime visualization
###########################################################################################################################
def limevisual(pData, pDesc, Idx, pClassNames, pAccountName, pVec, nNumFeatures, nTopLabels, tLabels, pRootDir):
    try:
        pIntent = pData['Intent'][int(Idx)]
        _, pModels = loadmodel(pRootDir, pAccountName, pIntent)
        pPipeModel = make_pipeline(pVec, pModels)
        tokenizer = lambda doc: re.compile(r"(?u)\b\w\w+\b").findall(doc)
        pExplainer = LimeTextExplainer(class_names = pClassNames, split_expression = tokenizer)
        pExplainText = pExplainer.explain_instance(pData[pDesc][int(Idx)], classifier_fn = pPipeModel.predict_proba, num_features = int(nNumFeatures), top_labels = int(nTopLabels))
        pExplainText.show_in_notebook(text = pData[pDesc][int(Idx)], labels = tLabels)
        pExplainText.save_to_file('C:\\Users\\tamohant\\Desktop\\Auto_synthesis_Training_data\\AutoSynthesisLite\\demo.html', labels=None, predict_proba=True, show_predicted_value=True)
    except Exception as e:
        print('*** ERROR[001]: Error in visualiation file of Limevisual function: ', sys.exc_info()[0],str(e))
        print(traceback.format_exc())
        return(-1)
    return(0)
    
###########################################################################################################################
# Author        : Tapas Mohanty  
# Modified      : 
# Reviewer      :
# Functionality : Explanation of prediction through eli5 visualization
###########################################################################################################################
def eli5visual(pData, pDesc, Idx, pAccountName, pVec, nTopKeywrd, pRootDir):
    try:
        for i in range(len(Idx)):
            if Idx[i] <= len(pData):
                pIntent = pData['Intent'][int(Idx[i])]
                _, pModels = loadmodel(pRootDir, pAccountName, pIntent)
                pPipeModel = make_pipeline(pVec, pModels)
                pTe = TextExplainer(random_state=42).fit( pData[pDesc][int(Idx[i])], pPipeModel.predict_proba)
                pExplanation = pTe.explain_prediction()
                pHtml = format_as_html(pExplanation, force_weights=False, include_styles=False, horizontal_layout=True, show_feature_values=False)
                savehtml(pRootDir, pHtml, Idx[i], pIntent)
            else:
                print("Please select valid Id")

    except Exception as e:
        print('*** ERROR[003]: Error in visualiation file of eil5visual function: ', sys.exc_info()[0],str(e))
        print(traceback.format_exc())
        return(-1)
    return(0)

###########################################################################################################################
# Author        : Tapas Mohanty  
# Modified      : 
# Reviewer      :
# Functionality : save html
########################################################################################################################### 
def savehtml(pRootDir, html, Idx, pIntent):
    try:
        import webbrowser
        path = os.path.abspath(pRootDir + '\\' + 'visualoutput' + '\\' + str(pIntent) + '__' + str(Idx) + '__' + '.html')
        url = 'file://' + path

        with open(path, 'w') as f:
            f.write(html)
        webbrowser.open(url)
    except Exception as e:
        print('*** ERROR[004]: Error in visualiation file of save html function: ', sys.exc_info()[0],str(e))
        print(traceback.format_exc())
        return(-1)
    return(0)