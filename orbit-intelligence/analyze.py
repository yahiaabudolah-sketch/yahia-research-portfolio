"""Reproduce the maritime nowcast experiment; no network required."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
P=Path(__file__).resolve().parent
source=P/'singapore_monthly.csv'
if not source.exists():
    raise SystemExit('Missing local singapore_monthly.csv. See README.md and input_template.csv for permitted data sources and schema.')
D=pd.read_csv(source,parse_dates=['month']).set_index('month').sort_index()
assert D.index.is_unique and len(D)==93 and D.notna().all().all()
assert (D.index==pd.date_range('2019-01-01','2026-09-01',freq='MS')).all()
# Adjust monthly container call counts for number of days.
D['container_calls_daily']=D.container_ship_calls/D.index.days_in_month
D['teu_daily']=D.throughput_thousand_teu/D.index.days_in_month
for c in ['teu_daily','container_calls_daily','all_ship_calls_daily_mean']:
    D[c+'_log_yoy']=np.log(D[c]/D[c].shift(12))
X=['container_calls_daily_log_yoy','all_ship_calls_daily_mean_log_yoy']
Y='teu_daily_log_yoy'
M=D.dropna(subset=X+[Y]); train=M.loc[:'2024-12-01']; test=M.loc['2025-01-01':].copy()
# Fit only pre-2025 observations. Year-over-year differences remove seasonality and reduce trend effects.
b=np.linalg.lstsq(np.c_[np.ones(len(train)),train[X]],train[Y],rcond=None)[0]
base=D.teu_daily.shift(12).loc[test.index]*test.index.days_in_month
pred=base*np.exp(np.c_[np.ones(len(test)),test[X]]@b)
test['seasonal_naive']=base; test['ais_nowcast']=pred
# A trend-only comparator uses the historical mean daily YoY growth estimated on train.
test['trend_only']=base*np.exp(train[Y].mean())
metrics={}
for model in ['seasonal_naive','trend_only','ais_nowcast']:
    e=test[model]-test.throughput_thousand_teu
    metrics[model]={'MAE_thousand_TEU':float(abs(e).mean()),'RMSE_thousand_TEU':float(np.sqrt((e**2).mean())),'MAPE_pct':float((abs(e)/test.throughput_thousand_teu).mean()*100)}
# A negative z score flags unusual lower traffic compared with the prior 12 months; past only.
r=D.container_calls_daily.shift(1).rolling(12,min_periods=12)
D['traffic_z']=(D.container_calls_daily-r.mean())/r.std(ddof=1)
D['alert_low_traffic']=D.traffic_z < -2
out={'n_months':len(D),'train_n':len(train),'test_n':len(test),'coefficients':dict(zip(['intercept']+X,map(float,b))),'metrics':metrics,'corr_levels':float(D.container_calls_daily.corr(D.teu_daily)),'corr_yoy':float(M[X[0]].corr(M[Y])),'latest_month':str(D.index[-1].date()),'latest_calls_yoy_pct':float((D.container_calls_daily.iloc[-1]/D.container_calls_daily.iloc[-13]-1)*100),'latest_teu_yoy_pct':float((D.teu_daily.iloc[-1]/D.teu_daily.iloc[-13]-1)*100),'latest_z':float(D.traffic_z.iloc[-1]),'alert_count':int(D.alert_low_traffic.sum()),'train_end':'2024-12-01'}
(P/'results.json').write_text(json.dumps(out,indent=2));D.join(test[['seasonal_naive','trend_only','ais_nowcast']]).to_csv(P/'analysis_output.csv');test[['throughput_thousand_teu','seasonal_naive','trend_only','ais_nowcast']].to_csv(P/'holdout_predictions.csv')
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white','axes.grid':True,'grid.alpha':.18})
fig,ax=plt.subplots(figsize=(9,3.7)); ax.plot(D.index,D.teu_daily/D.teu_daily.iloc[:12].mean()*100,label='Official TEU/day'); ax.plot(D.index,D.container_calls_daily/D.container_calls_daily.iloc[:12].mean()*100,label='AIS container calls/day'); ax.axvline(pd.Timestamp('2025-01-01'),ls='--',color='gray',label='Holdout starts'); ax.set_ylabel('Index (2019 average = 100)');ax.legend(ncol=3,loc='upper left',fontsize=8);fig.tight_layout();fig.savefig(P/'activity.png',dpi=160);plt.close(fig)
fig,ax=plt.subplots(figsize=(9,3.7)); ax.plot(test.index,test.throughput_thousand_teu,label='Official actual',lw=2);ax.plot(test.index,test.ais_nowcast,label='AIS nowcast');ax.plot(test.index,test.trend_only,label='Trend-only');ax.plot(test.index,test.seasonal_naive,label='Seasonal naive',ls='--');ax.set_ylabel('Thousand TEU per month');ax.legend(ncol=2,fontsize=8);fig.tight_layout();fig.savefig(P/'holdout.png',dpi=160);plt.close(fig)
# Fully offline dashboard (Plotly JS embedded).
import plotly.graph_objects as go
from plotly.subplots import make_subplots
fig=make_subplots(rows=3,cols=1,subplot_titles=('Official throughput and AIS traffic (2019 = 100)','Chronological holdout: actual vs three models','Traffic anomaly z-score (past 12 months)'),vertical_spacing=.09)
for c,name in [('teu_daily','Official throughput'),('container_calls_daily','Container ship calls'),('all_ship_calls_daily_mean','All-vessel calls')]:
    fig.add_trace(go.Scatter(x=D.index,y=D[c]/D[c].iloc[:12].mean()*100,name=name),row=1,col=1)
for c,name in [('throughput_thousand_teu','Actual'),('ais_nowcast','AIS nowcast'),('seasonal_naive','Seasonal naive'),('trend_only','Trend only')]:
    fig.add_trace(go.Scatter(x=test.index,y=test[c],name=name),row=2,col=1)
fig.add_trace(go.Scatter(x=D.index,y=D.traffic_z,name='Traffic z-score'),row=3,col=1);fig.add_hline(y=-2,line_dash='dash',row=3,col=1)
fig.update_yaxes(title_text='Indexed activity',row=1,col=1);fig.update_yaxes(title_text='Thousand TEU',row=2,col=1);fig.update_yaxes(title_text='Standard deviations',row=3,col=1)
fig.update_layout(title='ORBIT INTELLIGENCE | Singapore maritime activity',height=1000,template='plotly_white',legend=dict(orientation='h',y=-.07),margin=dict(l=75,r=45,t=100,b=110));fig.write_html(P/'dashboard.html',include_plotlyjs=True)
print(json.dumps(out,indent=2))
