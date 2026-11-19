"""Reproduce the historical comparisons and simplified cash scenarios in USDm."""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
P=Path(__file__).resolve().parent
h=pd.read_csv(P/'financial_inputs.csv'); l=pd.read_csv(P/'liquidity_inputs.csv');h['FCF']=h.operating_cash_flow_usdm-h.capex_usdm;h['operating_margin']=h.operating_profit_usdm/h.revenue_usdm;h['period']=h.groupby('company').cumcount()+1
latest=h.groupby('company',sort=False).tail(1).merge(l,on=['company','fiscal_end']);latest['liquid_resources']=latest.cash_usdm+latest.short_term_investments_usdm;latest['sensitivity_FCF']=latest.FCF-latest.ocf_adjustment_usdm
ass=[('Rocket Lab','Downside',.05,.02,.12,.28),('Rocket Lab','Base',.20,.07,.18,.22),('Rocket Lab','Upside',.30,.11,.24,.18),('Planet Labs','Downside',.04,.01,.10,.30),('Planet Labs','Base',.16,.05,.20,.22),('Planet Labs','Upside',.25,.08,.27,.17),('Intuitive Machines','Downside',-.05,0,.10,.24),('Intuitive Machines','Base',.10,.04,.16,.18),('Intuitive Machines','Upside',.22,.07,.23,.14)]
rows=[]
for name,scenario,g,delta,cap,cx in ass:
    d=latest[latest.company==name].iloc[0];rev=d.revenue_usdm; margin=(d.operating_cash_flow_usdm-d.ocf_adjustment_usdm)/rev;cash=d.liquid_resources
    for year in range(1,6):
        rev*=1+g;margin=min(cap,margin+delta);ocf=rev*margin;capex=rev*cx;fcf=ocf-capex;shock=d.known_cash_commitment_stress_usdm if year==1 else 0;gap=max(0,100-(cash+fcf-shock));end=cash+fcf-shock+gap;rows.append([name,scenario,year,rev,margin,ocf,capex,fcf,cash,shock,gap,end]);cash=end
out=pd.DataFrame(rows,columns=['company','scenario','year','revenue','ocf_margin','ocf','capex','fcf','opening_liquidity','cash_shock','external_cash_need','closing_liquidity']);out.to_csv(P/'scenario_results.csv',index=False);h.to_csv(P/'historical_metrics.csv',index=False);latest.to_csv(P/'latest_metrics.csv',index=False)
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white','axes.grid':True,'grid.alpha':.18})
fig,ax=plt.subplots(figsize=(9,3.4));
for name,d in h.groupby('company',sort=False): ax.plot(d.period,d.FCF,marker='o',label=name)
ax.set_xticks([1,2,3],['FY ending 2023/24','FY ending 2024/25','FY ending 2025/26']);ax.axhline(0,color='gray',lw=.8);ax.set_ylabel('Free cash flow (USDm)');ax.legend(ncol=3,fontsize=8);fig.tight_layout();fig.savefig(P/'historical_fcf.png',dpi=160);plt.close(fig)
pivot=out.groupby(['company','scenario'],sort=False).external_cash_need.sum().unstack();pivot=pivot.reindex(['Rocket Lab','Planet Labs','Intuitive Machines'])[['Downside','Base','Upside']];ax=pivot.plot(kind='bar',figsize=(9,3.4),rot=0);ax.set_ylabel('External cash required (USDm)');ax.legend(ncol=3,fontsize=8);ax.figure.tight_layout();ax.figure.savefig(P/'funding_scenarios.png',dpi=160);plt.close(ax.figure)
print(pivot.round(1)); print(latest[['company','revenue_usdm','FCF','sensitivity_FCF','liquid_resources']].to_string(index=False))
