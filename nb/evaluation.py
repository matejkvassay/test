from collections import Counter
from IPython.core.display import display
from IPython.core.display import HTML
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from math import sqrt
import numpy as np

def plot_history(model,title):
    loss = model.history.history['loss']
    val_loss = model.history.history['val_loss']
    epochs = range(1, len(loss) + 1)
    plt.plot(epochs, loss, color='red', label='MSE Train')
    plt.plot(epochs, val_loss, color='green', label='MSE Dev')
    plt.title(title)
    plt.xlabel('Epochs') 
    plt.ylabel('MSE')
    plt.legend()
    plt.show()

def rmse(y_true,y_pred):
    return sqrt(mean_squared_error(y_true,y_pred))

def rmse_partial(y_true, y_pred):
    all_scores=list(set(y_true))
    y_true_dict={x:[] for x in all_scores}
    y_pred_dict={x:[] for x in all_scores}
    for i, score in enumerate(y_true):
        y_true_dict[score].append(score)
        y_pred_dict[score].append(y_pred[i])
    return {score: rmse(y_true_dict[score],y_pred_dict[score]) for score in all_scores}
        
def rmse_partial_avg(y_true, y_pred):
    rmse_dict=rmse_partial(y_true,y_pred)
    values=list(rmse_dict.values())
    return np.mean(values), np.std(values)

def rmse_partial_max(y_true, y_pred):
    rmse_dict=rmse_partial(y_true,y_pred)
    values=list(rmse_dict.values())
    return np.max(values)

def rmse_report(y_true,y_pred,round_decimals=3, title='RMSE report'):
    def ar(x):
        return np.around(x,decimals=round_decimals)
    baseline=rmse(y_true,[5.0]*len(y_true))
    HTML_TEMPLATE="""
    <h2> {} </h2>
    <h3> RMSE </h3>
    <hr>
    <div>
        <table>
            <tr>
                <td>RMSE (baseline &forall;1.0)</td>
                <td>{}</td>
            </tr>
            <tr>
                <td>RMSE</td>
                <td>{}</td>
            </tr>
        </table>
    <hr>
    <h3> Partial RMSE </h3>
        <table>
            <tr>
                <td>Mean partial RMSE (baseline &forall;1.0)</td>
                <td>{}</td>
            </tr>
            <tr>
                <td>Max partial RMSE (baseline &forall;1.0)</td>
                <td>{}</td>
            </tr>
            <tr>
                <td>St.dev. partial RMSE (baseline &forall;1.0)</td>
                <td>{}</td>
            </tr>
            <tr>
                <td>Mean partial RMSE</td>
                <td><b>{}</b></td>
            </tr>
            <tr>
                <td>Max partial RMSE</td>
                <td>{}</td>
            </tr>
            <tr>
                <td>St.dev. partial RMSE</td>
                <td>{}</td>
            </tr>            
        </table>
    </div>
    <h3> Improvement over baseline (&forall;1.0) </h3>
    <hr>
    <div>
        <table>
            <tr>
                <td>RMSE</td>
                <td>{}</td>
            </tr>
            <tr>
                <td>Mean partial RMSE</td>
                <td><b>{}</b></td>
            </tr>
            <tr>
                <td>Max partial RMSE</td>
                <td>{}</td>
            </tr>
        </table>
    </div>
    
    <h3> Partial RMSE detailed</h3>
    <hr>
    <div>
        <table>
            <tr>
                <th>Review Score</th>
                <th>RMSE</th>
                <th>RMSE baseline (&forall;1.0)</th>
                <th>Improvement over baseline</th>
            </tr>
            {}
        </table>
    </div>
    """
    PARTIAL_ROW_TEMPLATE='''
    <tr>
        <td>
            {}
        </td>
        <td>
            {}
        </td>
        <td>
            {}
        </td>
        <td>
            {}
        </td>
    </tr>
    '''
    overall=rmse(y_true,y_pred)
    partial_avg,partial_std=rmse_partial_avg(y_true,y_pred)
    partial_max=rmse_partial_max(y_true,y_pred)
    partial_avg_baseline,partial_std_baseline=rmse_partial_avg(y_true,[5.0]*len(y_true))
    partial_max_baseline=rmse_partial_max(y_true,[5.0]*len(y_true))
    improvement_marmse=partial_avg_baseline-partial_avg
    improvement_rmse=baseline-overall
    improvement_rmse_partial_max=partial_max_baseline - partial_max
    
    #partial rows
    partial=rmse_partial(y_true,y_pred)
    partial_baseline=rmse_partial(y_true, [5.0]*len(y_true))
    partial=sorted(partial.items(),key=lambda x:x[0],reverse=True)
    partial_table_rows=[]
    for key,value in partial:
        value_baseline=partial_baseline[key]
        diff_baseline=value_baseline-value
        partial_table_rows.append(PARTIAL_ROW_TEMPLATE.format(key,ar(value),ar(value_baseline),ar(diff_baseline)))
    partial_table_rows='\n'.join(partial_table_rows)
    
    html=HTML_TEMPLATE.format(title,
                              ar(baseline),
                              ar(overall),
                              ar(partial_avg_baseline),
                              ar(partial_std_baseline),
                              ar(partial_max_baseline),
                              ar(partial_avg),
                              ar(partial_std),
                              ar(partial_max),
                              ar(improvement_rmse),
                              ar(improvement_marmse),
                              ar(improvement_rmse_partial_max),
                              partial_table_rows)
    display(HTML(html))