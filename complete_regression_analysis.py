import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson
import warnings
warnings.filterwarnings('ignore')

# ── Palette ──────────────────────────────────────────────────────────────────
BG    = '#0F1117'; CARD  = '#1A1D27'
A1='#7C83FD'; A2='#FD7C9E'; A3='#7CFDB0'; A4='#FDB97C'; A5='#C07CFD'
WHITE='#E8ECF4'; GRID='#2E3250'; MUTED='#8B92B8'

plt.rcParams.update({
    'figure.facecolor':BG,'axes.facecolor':CARD,'axes.edgecolor':GRID,
    'axes.labelcolor':WHITE,'text.color':WHITE,'xtick.color':MUTED,
    'ytick.color':MUTED,'grid.color':GRID,'grid.linewidth':0.6,
    'font.family':'monospace','axes.titlesize':12,'axes.labelsize':10,
    'axes.titlepad':10,
})

# ── Data ─────────────────────────────────────────────────────────────────────
df = pd.read_csv('F:\MASTER\Regression Analysis\Project\student_performance_dataset.csv')
le = LabelEncoder()
df['Parental_enc'] = le.fit_transform(df['Parental_Involvement'])

features = ['Hours_Studied','Attendance_Rate','Sleep_Hours','Previous_Score',
            'Tutoring_Support','Parental_enc','Extracurricular_Activities','Motivation_Level']
feat_labels = ['Hours Studied','Attendance Rate','Sleep Hours','Previous Score',
               'Tutoring','Parental Inv.','Extracurricular','Motivation']
target = 'Performance_Index'
X = df[features]; y = df[target]

scaler = StandardScaler()
X_sc = scaler.fit_transform(X)
X_tr, X_ts, y_tr, y_ts = train_test_split(X_sc, y, test_size=0.2, random_state=42)

lr    = LinearRegression().fit(X_tr, y_tr)
ridge = Ridge(alpha=1.0).fit(X_tr, y_tr)
lasso = Lasso(alpha=0.1).fit(X_tr, y_tr)

y_pred    = lr.predict(X_ts)
residuals = y_ts - y_pred

X_sm = sm.add_constant(X)
ols  = sm.OLS(y, X_sm).fit()

def met(m, Xtr, ytr, Xts, yts, name):
    return {'Model':name,
            'Train R²':r2_score(ytr,m.predict(Xtr)),
            'Test R²': r2_score(yts,m.predict(Xts)),
            'Train RMSE':np.sqrt(mean_squared_error(ytr,m.predict(Xtr))),
            'Test RMSE': np.sqrt(mean_squared_error(yts,m.predict(Xts))),
            'MAE':mean_absolute_error(yts,m.predict(Xts))}

res_df = pd.DataFrame([met(lr,X_tr,y_tr,X_ts,y_ts,'OLS'),
                        met(ridge,X_tr,y_tr,X_ts,y_ts,'Ridge'),
                        met(lasso,X_tr,y_tr,X_ts,y_ts,'Lasso')])

cv = cross_val_score(LinearRegression(), X_sc, y, cv=10, scoring='r2')

# VIF
X_vif = pd.DataFrame(X_sc, columns=features)
X_vif_c = sm.add_constant(X_vif)
vif_data = pd.DataFrame({
    'Feature': features,
    'VIF': [variance_inflation_factor(X_vif_c.values, i+1) for i in range(len(features))]
})

dw = durbin_watson(ols.resid)
jb_result = stats.jarque_bera(residuals); jb_stat, jb_p = jb_result[0], jb_result[1]

print("=== Model Results ===")
print(res_df.round(4).to_string(index=False))
print(f"\nCV Mean R² = {cv.mean():.4f} ± {cv.std():.4f}")
print(f"\nDurbin-Watson = {dw:.4f}")
print(f"Jarque-Bera stat = {jb_stat:.4f}, p = {jb_p:.4f}")
print(f"\nVIF:\n{vif_data.round(3).to_string(index=False)}")
print(f"\nOLS R²={ols.rsquared:.4f}, Adj R²={ols.rsquared_adj:.4f}")
print(f"F={ols.fvalue:.2f}, p={ols.f_pvalue:.2e}")

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — Enhanced EDA
# ═══════════════════════════════════════════════════════════════════════════════
fig1 = plt.figure(figsize=(22, 24), facecolor=BG)
fig1.suptitle('STUDENT PERFORMANCE  ─  Exploratory Data Analysis',
              fontsize=20, color=WHITE, fontweight='bold', y=0.99, fontfamily='monospace')
gs1 = gridspec.GridSpec(4, 3, figure=fig1, hspace=0.50, wspace=0.38,
                         left=0.06, right=0.97, top=0.96, bottom=0.03)

# 1-A Distribution
ax = fig1.add_subplot(gs1[0,0])
vals = y.values
ax.hist(vals, bins=32, color=A1, alpha=0.82, edgecolor=BG, lw=0.5)
ax.axvline(vals.mean(),  color=A2, lw=2.2, ls='--', label=f'Mean={vals.mean():.1f}')
ax.axvline(np.median(vals), color=A3, lw=2.2, ls='--', label=f'Median={np.median(vals):.1f}')
ax.set_title('Distribution of Performance Index', color=WHITE)
ax.set_xlabel('Performance Index'); ax.set_ylabel('Frequency')
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
# add skew/kurt
sk = stats.skew(vals); ku = stats.kurtosis(vals)
ax.text(0.97,0.95,f'Skew={sk:.2f}\nKurt={ku:.2f}',
        transform=ax.transAxes, ha='right', va='top', fontsize=8,
        bbox=dict(boxstyle='round,pad=0.3',facecolor=CARD,alpha=0.85))

# 1-B Correlation heatmap
ax2 = fig1.add_subplot(gs1[0,1:])
num_df = df[features+[target]].rename(columns=dict(zip(
    features+[target], feat_labels+['Performance'])))
corr = num_df.corr()
cmap = sns.diverging_palette(240,10,as_cmap=True)
sns.heatmap(corr, ax=ax2, cmap=cmap, center=0, annot=True, fmt='.2f',
            annot_kws={'size':8,'color':WHITE}, linewidths=0.5, linecolor=BG,
            cbar_kws={'shrink':0.8})
ax2.set_title('Correlation Matrix (Pearson)', color=WHITE)
ax2.tick_params(axis='x', rotation=35, labelsize=8)
ax2.tick_params(axis='y', rotation=0,  labelsize=8)

# 1-C to 1-G Scatter plots
scatters = [('Hours_Studied',A1,'Hours Studied'),
            ('Attendance_Rate',A2,'Attendance Rate'),
            ('Previous_Score',A3,'Previous Score'),
            ('Sleep_Hours',A4,'Sleep Hours'),
            ('Motivation_Level',A5,'Motivation Level')]
positions = [(1,0),(1,1),(1,2),(2,0),(2,1)]
for (feat,color,label), pos in zip(scatters, positions):
    ax = fig1.add_subplot(gs1[pos])
    ax.scatter(df[feat], y, color=color, alpha=0.35, s=18, edgecolors='none')
    m2,b2,r2,p2,_ = stats.linregress(df[feat], y)
    xl = np.linspace(df[feat].min(), df[feat].max(), 100)
    ax.plot(xl, m2*xl+b2, color=WHITE, lw=2)
    ax.set_title(f'{label} vs Performance', color=WHITE, fontsize=10)
    ax.set_xlabel(label, fontsize=9); ax.set_ylabel('Performance Index', fontsize=9)
    sig = '***' if p2<0.001 else '**' if p2<0.01 else '*' if p2<0.05 else 'ns'
    ax.text(0.05,0.92,f'r = {r2:.3f}  {sig}', transform=ax.transAxes,
            fontsize=9, color=WHITE,
            bbox=dict(boxstyle='round,pad=0.3',facecolor=CARD,alpha=0.85))
    ax.grid(True, alpha=0.3)

# 1-H Boxplot parental
ax_bp = fig1.add_subplot(gs1[2,2])
grps = [df[df['Parental_Involvement']==g][target].values for g in ['Low','Medium','High']]
bp = ax_bp.boxplot(grps, patch_artist=True, notch=True,
                   medianprops=dict(color=WHITE,lw=2.2),
                   whiskerprops=dict(color=MUTED,lw=1.3),
                   capprops=dict(color=MUTED,lw=1.3),
                   flierprops=dict(marker='o',color=A2,markersize=3,alpha=0.5))
for patch,c in zip(bp['boxes'],[A1,A3,A4]):
    patch.set_facecolor(c); patch.set_alpha(0.7)
ax_bp.set_xticklabels(['Low','Medium','High'])
ax_bp.set_title('Performance by Parental Involvement', color=WHITE, fontsize=10)
ax_bp.set_ylabel('Performance Index'); ax_bp.grid(True, alpha=0.3)
# ANOVA
f_stat, f_p = stats.f_oneway(*grps)
ax_bp.text(0.05,0.94,f'ANOVA p={f_p:.3f}',transform=ax_bp.transAxes,
           fontsize=8,bbox=dict(boxstyle='round,pad=0.3',facecolor=CARD,alpha=0.85))

# 1-I Bar tutoring
ax_bar = fig1.add_subplot(gs1[3,0])
tut_means = df.groupby('Tutoring_Support')[target].mean()
tut_std   = df.groupby('Tutoring_Support')[target].std()
bars = ax_bar.bar(['No Tutoring','Tutoring'], tut_means.values,
                  color=[A2,A3], alpha=0.82, width=0.5,
                  yerr=tut_std.values, capsize=5,
                  error_kw=dict(ecolor=WHITE,elinewidth=1.5),
                  edgecolor=BG)
for b,v in zip(bars,tut_means.values):
    ax_bar.text(b.get_x()+b.get_width()/2, v+1.5, f'{v:.1f}',
                ha='center', color=WHITE, fontsize=10, fontweight='bold')
ax_bar.set_title('Avg Performance: Tutoring (±1 SD)', color=WHITE, fontsize=10)
ax_bar.set_ylabel('Mean Performance Index')
ax_bar.set_ylim(0, tut_means.max()*1.3); ax_bar.grid(True,alpha=0.3,axis='y')
t_stat,t_p = stats.ttest_ind(df[df['Tutoring_Support']==1][target],
                               df[df['Tutoring_Support']==0][target])
ax_bar.text(0.5,0.92,f't-test p={t_p:.4f}',transform=ax_bar.transAxes,
            ha='center',fontsize=8,bbox=dict(boxstyle='round,pad=0.3',facecolor=CARD,alpha=0.85))

# 1-J Extracurricular violin
ax_vi = fig1.add_subplot(gs1[3,1])
g0 = df[df['Extracurricular_Activities']==0][target].values
g1 = df[df['Extracurricular_Activities']==1][target].values
vp = ax_vi.violinplot([g0,g1], positions=[0,1], showmedians=True, showextrema=True)
for body,c in zip(vp['bodies'],[A2,A1]):
    body.set_facecolor(c); body.set_alpha(0.6)
vp['cmedians'].set_color(WHITE); vp['cmedians'].set_lw(2)
ax_vi.set_xticks([0,1]); ax_vi.set_xticklabels(['No Extra','Extracurricular'])
ax_vi.set_title('Performance by Extracurricular', color=WHITE, fontsize=10)
ax_vi.set_ylabel('Performance Index'); ax_vi.grid(True,alpha=0.3)

# 1-K Descriptive stats table
ax_tbl = fig1.add_subplot(gs1[3,2])
ax_tbl.axis('off')
desc = df[['Hours_Studied','Attendance_Rate','Sleep_Hours',
           'Previous_Score','Motivation_Level','Performance_Index']].describe().round(2)
short = ['Hrs Study','Attend %','Sleep Hrs','Prev Score','Motivation','Performance']
tbl = ax_tbl.table(cellText=desc.values.tolist(), rowLabels=desc.index.tolist(),
                    colLabels=short, cellLoc='center', loc='center')
tbl.auto_set_font_size(False); tbl.set_fontsize(7.5)
for (r,c),cell in tbl.get_celld().items():
    cell.set_facecolor('#252840' if r%2==0 else CARD)
    cell.set_edgecolor(GRID); cell.set_text_props(color=WHITE)
    if r==0: cell.set_facecolor('#3A3F6E'); cell.set_text_props(color=A1,fontweight='bold')
    if c==-1: cell.set_facecolor('#3A3F6E'); cell.set_text_props(color=A4)
ax_tbl.set_title('Descriptive Statistics', color=WHITE, fontsize=10, pad=8)

plt.savefig(r'F:\MASTER\Regression Analysis\Project\fig1_eda_v2.png', dpi=130, bbox_inches='tight', facecolor=BG)
plt.close()
print("Fig 1 saved ✓")

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Regression Results
# ═══════════════════════════════════════════════════════════════════════════════
fig2 = plt.figure(figsize=(22, 24), facecolor=BG)
fig2.suptitle('STUDENT PERFORMANCE  ─  Regression Analysis Results',
              fontsize=20, color=WHITE, fontweight='bold', y=0.99, fontfamily='monospace')
gs2 = gridspec.GridSpec(4, 3, figure=fig2, hspace=0.50, wspace=0.38,
                         left=0.06, right=0.97, top=0.96, bottom=0.03)

# 2-A Model R² comparison
ax = fig2.add_subplot(gs2[0,:2])
x_pos = np.arange(3); w = 0.32
b1 = ax.bar(x_pos-w/2, res_df['Train R²'], w, color=A1, alpha=0.85, label='Train R²')
b2 = ax.bar(x_pos+w/2, res_df['Test R²'],  w, color=A2, alpha=0.85, label='Test R²')
for b,v in zip(b1,res_df['Train R²']): ax.text(b.get_x()+b.get_width()/2,v+0.005,f'{v:.4f}',ha='center',color=WHITE,fontsize=9)
for b,v in zip(b2,res_df['Test R²']):  ax.text(b.get_x()+b.get_width()/2,v+0.005,f'{v:.4f}',ha='center',color=WHITE,fontsize=9)
ax.set_xticks(x_pos); ax.set_xticklabels(res_df['Model'])
ax.set_title('Model Comparison — R² Score', color=WHITE)
ax.set_ylabel('R²'); ax.set_ylim(0,1.08)
ax.legend(); ax.grid(True,alpha=0.3,axis='y')
ax.axhline(0.9, color=A4, lw=1.5, ls=':', alpha=0.7)
ax.text(2.48, 0.91, 'R²=0.90', color=A4, fontsize=8)

# 2-B RMSE + MAE
ax = fig2.add_subplot(gs2[0,2])
ax.bar(x_pos-w/2, res_df['Test RMSE'], w, color=A3, alpha=0.85, label='Test RMSE')
ax.bar(x_pos+w/2, res_df['MAE'],       w, color=A4, alpha=0.85, label='MAE')
for b,v in zip(ax.patches[:3],res_df['Test RMSE']): ax.text(b.get_x()+b.get_width()/2,v+0.03,f'{v:.2f}',ha='center',color=WHITE,fontsize=8)
ax.set_xticks(x_pos); ax.set_xticklabels(res_df['Model'],fontsize=9)
ax.set_title('Test RMSE vs MAE', color=WHITE)
ax.set_ylabel('Error (points)'); ax.legend(fontsize=8); ax.grid(True,alpha=0.3,axis='y')

# 2-C Actual vs Predicted
ax = fig2.add_subplot(gs2[1,0])
ax.scatter(y_ts, y_pred, color=A1, alpha=0.45, s=22, edgecolors='none')
mn,mx = y_ts.min(), y_ts.max()
ax.plot([mn,mx],[mn,mx], color=A2, lw=2.2, ls='--', label='Perfect fit')
m3,b3,r3,_,_ = stats.linregress(y_ts, y_pred)
xl = np.linspace(mn,mx,100)
ax.plot(xl, m3*xl+b3, color=A3, lw=1.8, ls='-', label='OLS fit')
ax.set_title('Actual vs Predicted (OLS)', color=WHITE)
ax.set_xlabel('Actual'); ax.set_ylabel('Predicted')
r2v = r2_score(y_ts, y_pred)
ax.text(0.05,0.92,f'R² = {r2v:.4f}', transform=ax.transAxes, color=WHITE, fontsize=10,
        bbox=dict(boxstyle='round,pad=0.3',facecolor=CARD,alpha=0.85))
ax.legend(fontsize=8); ax.grid(True,alpha=0.3)

# 2-D Residuals vs Fitted
ax = fig2.add_subplot(gs2[1,1])
ax.scatter(y_pred, residuals, color=A3, alpha=0.45, s=22, edgecolors='none')
ax.axhline(0, color=A2, lw=2.2, ls='--')
# LOWESS smoother
from statsmodels.nonparametric.smoothers_lowess import lowess
lw_out = lowess(residuals, y_pred, frac=0.3)
ax.plot(lw_out[:,0], lw_out[:,1], color=A4, lw=2.5, label='LOWESS')
ax.set_title('Residuals vs Fitted Values', color=WHITE)
ax.set_xlabel('Fitted Values'); ax.set_ylabel('Residuals')
ax.legend(fontsize=8); ax.grid(True,alpha=0.3)

# 2-E Residual distribution
ax = fig2.add_subplot(gs2[1,2])
ax.hist(residuals, bins=30, color=A1, alpha=0.8, edgecolor=BG, density=True)
xr = np.linspace(residuals.min(), residuals.max(), 200)
ax.plot(xr, stats.norm.pdf(xr, residuals.mean(), residuals.std()),
        color=A2, lw=2.5, label='Normal fit')
ax.axvline(0, color=A4, lw=1.5, ls='--')
ax.set_title('Residual Distribution', color=WHITE)
ax.set_xlabel('Residuals'); ax.set_ylabel('Density')
ax.text(0.97,0.92,f'JB p={jb_p:.3f}', transform=ax.transAxes, ha='right',
        fontsize=8, bbox=dict(boxstyle='round,pad=0.3',facecolor=CARD,alpha=0.85))
ax.legend(fontsize=8); ax.grid(True,alpha=0.3)

# 2-F Q-Q Plot
ax = fig2.add_subplot(gs2[2,0])
(osm,osr),(slope,intercept,r_qq) = stats.probplot(residuals, dist='norm')
ax.scatter(osm, osr, color=A1, alpha=0.5, s=20)
ax.plot(osm, slope*np.array(osm)+intercept, color=A2, lw=2.2)
ax.fill_between(osm,
                slope*np.array(osm)+intercept - 1.96*np.std(osr-slope*np.array(osm)-intercept),
                slope*np.array(osm)+intercept + 1.96*np.std(osr-slope*np.array(osm)-intercept),
                alpha=0.15, color=A2)
ax.set_title('Q-Q Plot — Normality of Residuals', color=WHITE)
ax.set_xlabel('Theoretical Quantiles'); ax.set_ylabel('Sample Quantiles')
ax.text(0.05,0.92,f'r = {r_qq:.4f}', transform=ax.transAxes, fontsize=9,
        bbox=dict(boxstyle='round,pad=0.3',facecolor=CARD,alpha=0.85))
ax.grid(True,alpha=0.3)

# 2-G Feature Coefficients
ax = fig2.add_subplot(gs2[2,1:])
coefs = lr.coef_
sidx  = np.argsort(np.abs(coefs))[::-1]
c_colors = [A1 if c>=0 else A2 for c in coefs[sidx]]
bars = ax.barh(range(len(feat_labels)), coefs[sidx],
               color=c_colors, alpha=0.85, edgecolor=BG, height=0.6)
ax.set_yticks(range(len(feat_labels)))
ax.set_yticklabels([feat_labels[i] for i in sidx])
ax.axvline(0, color=WHITE, lw=1, alpha=0.5)
for i,(b,v) in enumerate(zip(bars, coefs[sidx])):
    ax.text(v + (0.05 if v>=0 else -0.05), i, f'{v:.3f}',
            va='center', ha='left' if v>=0 else 'right',
            color=WHITE, fontsize=8)
ax.set_title('Standardized Regression Coefficients (OLS)', color=WHITE)
ax.set_xlabel('Coefficient Value')
ax.grid(True,alpha=0.3,axis='x')
pp = mpatches.Patch(color=A1,label='Positive'); np_ = mpatches.Patch(color=A2,label='Negative')
ax.legend(handles=[pp,np_], fontsize=9)

# 2-H Cross-validation
ax = fig2.add_subplot(gs2[3,0])
ax.plot(range(1,11), cv, marker='o', color=A1, lw=2, markersize=8, markerfacecolor=A2)
ax.axhline(cv.mean(), color=A4, ls='--', lw=2, label=f'Mean={cv.mean():.4f}')
ax.fill_between(range(1,11), cv.mean()-cv.std(), cv.mean()+cv.std(),
                alpha=0.18, color=A1)
ax.set_title('10-Fold Cross Validation R²', color=WHITE)
ax.set_xlabel('Fold'); ax.set_ylabel('R²'); ax.set_xticks(range(1,11))
ax.text(0.97,0.08,f'SD={cv.std():.4f}', transform=ax.transAxes, ha='right',
        fontsize=8, bbox=dict(boxstyle='round,pad=0.3',facecolor=CARD,alpha=0.85))
ax.legend(fontsize=9); ax.grid(True,alpha=0.3)

# 2-I VIF Chart
ax = fig2.add_subplot(gs2[3,1])
vif_sorted = vif_data.sort_values('VIF', ascending=True)
vif_colors = [A2 if v>10 else A4 if v>5 else A3 for v in vif_sorted['VIF']]
ax.barh(range(len(vif_sorted)), vif_sorted['VIF'], color=vif_colors, alpha=0.85, edgecolor=BG)
ax.set_yticks(range(len(vif_sorted)))
ax.set_yticklabels([feat_labels[features.index(f)] for f in vif_sorted['Feature']])
ax.axvline(5,  color=A4, lw=1.5, ls='--', alpha=0.8, label='VIF=5 (moderate)')
ax.axvline(10, color=A2, lw=1.5, ls='--', alpha=0.8, label='VIF=10 (high)')
ax.set_title('Multicollinearity — VIF', color=WHITE)
ax.set_xlabel('Variance Inflation Factor')
ax.legend(fontsize=8); ax.grid(True,alpha=0.3,axis='x')
for i,v in enumerate(vif_sorted['VIF']):
    ax.text(v+0.02, i, f'{v:.2f}', va='center', color=WHITE, fontsize=8)

# 2-J p-values significance
ax = fig2.add_subplot(gs2[3,2])
pvals = ols.pvalues[1:]
pv_colors = [A3 if p<0.001 else A4 if p<0.05 else A2 for p in pvals]
ax.barh(feat_labels, -np.log10(pvals+1e-300), color=pv_colors, alpha=0.85, edgecolor=BG)
ax.axvline(-np.log10(0.05),  color=A4, lw=1.8, ls='--', label='p=0.05')
ax.axvline(-np.log10(0.001), color=A3, lw=1.8, ls='--', label='p=0.001')
ax.set_title('Feature Significance (−log₁₀ p-value)', color=WHITE)
ax.set_xlabel('−log₁₀(p-value)')
s1 = mpatches.Patch(color=A3,label='p<0.001 ***')
s2 = mpatches.Patch(color=A4,label='p<0.05 *')
s3 = mpatches.Patch(color=A2,label='p≥0.05 ns')
ax.legend(handles=[s1,s2,s3], fontsize=8); ax.grid(True,alpha=0.3,axis='x')

plt.savefig('/home/claude/fig2_regression_v2.png', dpi=130, bbox_inches='tight', facecolor=BG)
plt.close()
print("Fig 2 saved ✓")

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — Outlier + Leverage + Cook's Distance + Scale-Location + Ridge Path
# ═══════════════════════════════════════════════════════════════════════════════
fig3 = plt.figure(figsize=(22, 16), facecolor=BG)
fig3.suptitle('STUDENT PERFORMANCE  ─  Advanced Diagnostics',
              fontsize=20, color=WHITE, fontweight='bold', y=0.99, fontfamily='monospace')
gs3 = gridspec.GridSpec(2, 3, figure=fig3, hspace=0.45, wspace=0.38,
                         left=0.06, right=0.97, top=0.95, bottom=0.05)

# 3-A Cook's Distance
ax = fig3.add_subplot(gs3[0,0:2])
influence = ols.get_influence()
cooks_d   = influence.cooks_distance[0]
thresh    = 4 / len(y)
colors_c  = [A2 if c > thresh else A1 for c in cooks_d]
ax.stem(range(len(cooks_d)), cooks_d, linefmt='-', markerfmt='o',
        basefmt=' ')


outlier_idx = np.where(cooks_d > thresh)[0]
ax.scatter(outlier_idx, cooks_d[outlier_idx], color=A2, s=40, zorder=5, label=f'Influential ({len(outlier_idx)} pts)')
ax.axhline(thresh, color=A4, lw=2, ls='--', label=f"Threshold = 4/n = {thresh:.4f}")
ax.set_title("Cook's Distance — Influential Observations", color=WHITE)
ax.set_xlabel('Observation Index'); ax.set_ylabel("Cook's D")
ax.legend(fontsize=9); ax.grid(True,alpha=0.3)

# 3-B Leverage vs Standardized Residuals (Influence plot)
ax = fig3.add_subplot(gs3[0,2])
leverage    = influence.hat_matrix_diag
std_resid   = influence.resid_studentized_internal
ax.scatter(leverage, std_resid, color=A1, alpha=0.4, s=22, edgecolors='none')
high_lev    = leverage > 2*(len(features)+1)/len(y)
high_res    = np.abs(std_resid) > 2
both        = high_lev & high_res
ax.scatter(leverage[both], std_resid[both], color=A2, s=50, zorder=5, label='High influence')
ax.axhline( 2, color=A4, lw=1.5, ls='--', alpha=0.7)
ax.axhline(-2, color=A4, lw=1.5, ls='--', alpha=0.7, label='±2 std resid')
ax.axvline(2*(len(features)+1)/len(y), color=A3, lw=1.5, ls='--', alpha=0.7, label='Leverage threshold')
ax.set_title('Leverage vs Standardized Residuals', color=WHITE)
ax.set_xlabel('Leverage (Hat values)'); ax.set_ylabel('Standardized Residuals')
ax.legend(fontsize=7); ax.grid(True,alpha=0.3)

# 3-C Scale-Location plot
ax = fig3.add_subplot(gs3[1,0])
sqrt_std = np.sqrt(np.abs(std_resid))
ols_fitted = ols.fittedvalues; ax.scatter(ols_fitted, np.sqrt(np.abs(influence.resid_studentized_internal)), color=A3, alpha=0.4, s=22, edgecolors='none')
lw2 = lowess(np.sqrt(np.abs(influence.resid_studentized_internal)), ols_fitted, frac=0.4)
ax.plot(lw2[:,0], lw2[:,1], color=A2, lw=2.5, label='LOWESS')
ax.set_title('Scale-Location (Homoscedasticity)', color=WHITE)
ax.set_xlabel('Fitted Values'); ax.set_ylabel('√|Standardized Residuals|')
ax.legend(fontsize=8); ax.grid(True,alpha=0.3)

# 3-D Ridge Regularization Path
ax = fig3.add_subplot(gs3[1,1])
alphas = np.logspace(-3, 4, 100)
ridge_coefs = []
for a in alphas:
    r = Ridge(alpha=a).fit(X_tr, y_tr)
    ridge_coefs.append(r.coef_)
ridge_coefs = np.array(ridge_coefs)
colors_ridge = [A1,A2,A3,A4,A5,'#FD7C7C','#7CB8FD','#FDFD7C']
for i,c in enumerate(colors_ridge):
    ax.plot(np.log10(alphas), ridge_coefs[:,i], color=c, lw=1.8, label=feat_labels[i])
ax.axvline(0, color=WHITE, lw=1.2, ls='--', alpha=0.5, label='alpha=1')
ax.set_title('Ridge Regularization Path', color=WHITE)
ax.set_xlabel('log₁₀(alpha)'); ax.set_ylabel('Coefficient Value')
ax.legend(fontsize=7, loc='upper right'); ax.grid(True,alpha=0.3)

# 3-E Prediction Error Pie
ax = fig3.add_subplot(gs3[1,2])
abs_e = np.abs(y_ts - y_pred)
bins_e = [0,3,6,10,15,100]
lbls_e = ['0-3 pts','3-6 pts','6-10 pts','10-15 pts','>15 pts']
counts,_ = np.histogram(abs_e, bins=bins_e)
pie_colors = [A3,A1,A4,A2,'#FF3333']
wedges,texts,autotexts = ax.pie(counts, labels=lbls_e, autopct='%1.1f%%',
    colors=pie_colors, startangle=90,
    textprops={'color':WHITE,'fontsize':9},
    wedgeprops={'edgecolor':BG,'linewidth':1.5})
for at in autotexts: at.set_fontsize(8)
ax.set_title('Prediction Error Distribution', color=WHITE)
ax.text(0,-1.4,f'Mean Abs Error = {abs_e.mean():.2f} pts',
        ha='center',color=MUTED,fontsize=9)

plt.savefig('/home/claude/fig3_diagnostics_v2.png', dpi=130, bbox_inches='tight', facecolor=BG)
plt.close()
print("Fig 3 saved ✓")
print("\n✅ All 3 figures complete!")
print(f"VIF max = {vif_data['VIF'].max():.2f} — {'✅ No multicollinearity' if vif_data['VIF'].max() < 5 else '⚠️ Check VIF'}")
print(f"DW = {dw:.4f} — {'✅ No autocorrelation' if 1.5 < dw < 2.5 else '⚠️ Check autocorrelation'}")
print(f"JB p = {jb_p:.4f} — {'✅ Residuals normal' if jb_p > 0.05 else '⚠️ Non-normal residuals'}")
