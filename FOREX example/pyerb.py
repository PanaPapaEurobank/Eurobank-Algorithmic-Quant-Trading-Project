from datetime import datetime
import pandas as pd, numpy as np, blpapi, sqlite3, time, matplotlib.pyplot as plt, itertools, types, multiprocessing, ta, sqlite3, xlrd
from tqdm import tqdm
from sklearn import (manifold, datasets, decomposition, ensemble, discriminant_analysis, random_projection, neighbors)
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import norm, t
from scipy.stats import skew, kurtosis, entropy
from scipy.linalg import svd
from sklearn import linear_model
from sklearn.decomposition import PCA
from itertools import combinations
from ta.volume import *
from optparse import OptionParser

class pyerb:

    # Math Operators
    def d(df, **kwargs):
        if 'nperiods' in kwargs:
            nperiods = kwargs['nperiods']
        else:
            nperiods = 1
        out = df.diff(nperiods)
        return out

    def dlog(df, **kwargs):
        if 'nperiods' in kwargs:
            nperiods = kwargs['nperiods']
        else:
            nperiods = 1
        if 'fillna' in kwargs:
            fillna = kwargs['fillna']
        else:
            fillna = "yes"

        out = pyerb.d(np.log(df), nperiods=nperiods)

        if fillna == "yes":
            out = out.fillna(0)
        return out

    def r(df, **kwargs):
        if 'calcMethod' in kwargs:
            calcMethod = kwargs['calcMethod']
        else:
            calcMethod = 'Continuous'
        if 'nperiods' in kwargs:
            nperiods = kwargs['nperiods']
        else:
            nperiods = 1
        if 'fillna' in kwargs:
            fillna = kwargs['fillna']
        else:
            fillna = "yes"

        if calcMethod == 'Continuous':
            out = pyerb.d(np.log(df), nperiods=nperiods)
        elif calcMethod == 'Discrete':
            out = df.pct_change(nperiods)
        if calcMethod == 'Linear':
            diffDF = pyerb.d(df, nperiods=nperiods)
            out = diffDF.divide(df.iloc[0])

        if fillna == "yes":
            out = out.fillna(0)
        return out

    def E(df):
        out = df.mean(axis=1)
        return out

    def rs(df):
        out = df.sum(axis=1)
        return out

    def ew(df):
        out = np.log(pyerb.E(np.exp(df)))
        return out

    def cs(df):
        out = df.cumsum()
        return out

    def ecs(df):
        out = np.exp(df.cumsum())
        return out

    def pb(df):
        out = np.log(pyerb.rs(np.exp(df)))
        return out

    def sign(df):
        #df[df > 0] = 1
        #df[df < 0] = -1
        #df[df == 0] = 0
        out = np.sign(df)
        return out

    def S(df, **kwargs):

        if 'nperiods' in kwargs:
            nperiods = kwargs['nperiods']
        else:
            nperiods = 1

        out = df.shift(periods=nperiods)

        return out

    def fd(df):
        out = df.replace([np.inf, -np.inf], 0)
        return out

    def svd_flip(u, v, u_based_decision=True):
        """Sign correction to ensure deterministic output from SVD.
        Adjusts the columns of u and the rows of v such that the loadings in the
        columns in u that are largest in absolute value are always positive.
        Parameters
        ----------
        u : ndarray
            u and v are the output of `linalg.svd` or
            :func:`~sklearn.utils.extmath.randomized_svd`, with matching inner
            dimensions so one can compute `np.dot(u * s, v)`.
        v : ndarray
            u and v are the output of `linalg.svd` or
            :func:`~sklearn.utils.extmath.randomized_svd`, with matching inner
            dimensions so one can compute `np.dot(u * s, v)`.
        u_based_decision : boolean, (default=True)
            If True, use the columns of u as the basis for sign flipping.
            Otherwise, use the rows of v. The choice of which variable to base the
            decision on is generally algorithm dependent.
        Returns
        -------
        u_adjusted, v_adjusted : arrays with the same dimensions as the input.
        """
        if u_based_decision:
            # columns of u, rows of v
            max_abs_cols = np.argmax(np.abs(u), axis=0)
            signs = np.sign(u[max_abs_cols, range(u.shape[1])])
            u *= signs
            v *= signs[:, np.newaxis]
        else:
            # rows of v, columns of u
            max_abs_rows = np.argmax(np.abs(v), axis=1)
            signs = np.sign(v[range(v.shape[0]), max_abs_rows])
            u *= signs
            v *= signs[:, np.newaxis]
        return u, v

    def rowStoch(df):
        out = df.div(df.abs().sum(axis=1), axis=0)
        return out

    ############

    def roller(df, func, n):
        ROLL = df.rolling(window=n, center=False).apply(lambda x: func(x), raw=True)
        return ROLL

    def gapify(df, **kwargs):
        if 'steps' in kwargs:
            steps = kwargs['steps']
        else:
            steps = 5

        gapifiedDF = pd.DataFrame(np.nan, index=df.index, columns=df.columns)
        gapifiedDF.iloc[::steps, :] = df.iloc[::steps, :]

        gapifiedDF = gapifiedDF.ffill()

        return gapifiedDF

    def expander(df, func, n):
        EXPAND = df.expanding(min_periods=n, center=False).apply(lambda x: func(x))
        return EXPAND

    # Other operators (file readers, chunks etc.)

    def read_date(date):
        return xlrd.xldate.xldate_as_datetime(date, 0)

    def chunkReader(name):
        df = pd.read_csv(name, delimiter=';', chunksize=10000)
        return df

    # Quantitative Finance

    def sharpe(df):
        return df.mean() / df.std()

    def drawdown(pnl):
        """
        calculate max drawdown and duration
        Input:
            pnl, in $
        Returns:
            drawdown : vector of drawdwon values
            duration : vector of drawdown duration
        """
        cumret = pnl.cumsum()

        highwatermark = [0]

        idx = pnl.index
        drawdown = pd.Series(index=idx)
        drawdowndur = pd.Series(index=idx)

        for t in range(1, len(idx)):
            highwatermark.append(max(highwatermark[t - 1], cumret[t]))
            drawdown[t] = (highwatermark[t] - cumret[t])
            drawdowndur[t] = (0 if drawdown[t] == 0 else drawdowndur[t - 1] + 1)

        return drawdown, drawdowndur

    def rollNormalise(df, **kwargs):
        if 'mode' in kwargs:
            mode = kwargs['mode']
        else:
            mode = 'standardEMA'
        if 'nIn' in kwargs:
            nIn = kwargs['nIn']
        else:
            nIn = 25

        if mode == 'standardEMA':
            rollNormaliserDF = pyerb.ema(df, nperiods=nIn) / pyerb.rollVol(df, nIn=nIn)
        return rollNormaliserDF

    def rollStatistics(df, method, **kwargs):
        if 'nIn' in kwargs:
            nIn = kwargs['nIn']
        else:
            nIn = 25
        if 'alpha' in kwargs:
            alpha = kwargs['alpha']
        else:
            alpha = 0.01

        if method == 'Vol':
            rollStatisticDF = pyerb.roller(df, np.std, nIn)
        elif method == 'Skewness':
            rollStatisticDF = pyerb.roller(df, skew, nIn)
        elif method == 'Kurtosis':
            rollStatisticDF = pyerb.roller(df, kurtosis, nIn)
        elif method == 'VAR':
            rollStatisticDF = norm.ppf(1 - alpha) * pyerb.rollVol(df, nIn=nIn) - pyerb.ema(df, nperiods=nIn)
        elif method == 'CVAR':
            rollStatisticDF = alpha ** -1 * norm.pdf(norm.ppf(alpha)) * pyerb.rollVol(df, nIn=nIn) - pyerb.ema(df, nperiods=nIn)
        elif method == 'Sharpe':
            rollStatisticDF = pyerb.roller(df, pyerb.sharpe, nIn)

        return rollStatisticDF

    def maxDD_DF(df):
        ddList = []
        for j in df:
            df0 = pyerb.d(df[j])
            maxDD = pyerb.drawdown(df0)[0].max()
            ddList.append([j, -maxDD])

        ddDF = pd.DataFrame(ddList)
        ddDF.columns = ['Strategy', 'maxDD']
        ddDF = ddDF.set_index('Strategy')
        return ddDF

    def profitRatio(pnl):
        '''
        calculate profit ratio as sum(pnl)/drawdown
        Input: pnl  - daily pnl, Series or DataFrame
        '''

        def processVector(pnl):  # process a single column
            s = pnl.fillna(0)
            dd = pyerb.drawdown(s)[0]
            p = s.sum() / dd.max()
            return p

        if isinstance(pnl, pd.Series):
            return processVector(pnl)

        elif isinstance(pnl, pd.DataFrame):

            p = pd.Series(index=pnl.columns)

            for col in pnl.columns:
                p[col] = processVector(pnl[col])

            return p
        else:
            raise TypeError("Input must be DataFrame or Series, not " + str(type(pnl)))

    # Technical Analysis Operators

    def sma(df, **kwargs):
        if 'nperiods' in kwargs:
            nperiods = kwargs['nperiods']
        else:
            nperiods = 3
        SMA = df.rolling(nperiods).mean().fillna(0)
        return SMA

    def ema(df, **kwargs):
        if 'nperiods' in kwargs:
            nperiods = kwargs['nperiods']
        else:
            nperiods = 3
        EMA = df.ewm(span=nperiods, min_periods=nperiods).mean().fillna(0)
        return EMA

    def bb(df, **kwargs):
        if 'nperiods' in kwargs:
            nperiods = kwargs['nperiods']
        else:
            nperiods = 3
        if 'no_of_std' in kwargs:
            no_of_std = kwargs['no_of_std']
        else:
            no_of_std = 2
        dfBB = pd.DataFrame(df)
        dfBB['Price'] = df
        dfBB['rolling_mean'] = pyerb.ema(df, nperiods=nperiods)
        dfBB['rolling_std'] = df.rolling(nperiods).std()

        dfBB['MIDDLE'] = dfBB['rolling_mean']
        dfBB['UPPER'] = dfBB['MIDDLE'] + (dfBB['rolling_std'] * no_of_std)
        dfBB['LOWER'] = dfBB['MIDDLE'] - (dfBB['rolling_std'] * no_of_std)

        return dfBB[['Price', 'UPPER', 'MIDDLE', 'LOWER']]

    def rsi(df, n):
        i = 0
        UpI = [0]
        DoI = [0]
        while i + 1 < len(df):
            UpMove = df.loc[i + 1, 'High'] - df.loc[i, 'High']
            DoMove = df.loc[i, 'Low'] - df.loc[i + 1, 'Low']
            if UpMove > DoMove and UpMove > 0:
                UpD = UpMove
            else:
                UpD = 0
            UpI.append(UpD)
            if DoMove > UpMove and DoMove > 0:
                DoD = DoMove
            else:
                DoD = 0
            DoI.append(DoD)
            i = i + 1
        UpI = pd.Series(UpI)
        DoI = pd.Series(DoI)
        PosDI = pd.Series(UpI.ewm(span=n, min_periods=n).mean())
        NegDI = pd.Series(DoI.ewm(span=n, min_periods=n).mean())
        RSI = pd.DataFrame(PosDI / (PosDI + NegDI))
        RSI.columns = ['RSI']
        return RSI

    # Signals
    def sbb(df, **kwargs):
        if 'nperiods' in kwargs:
            nperiods = kwargs['nperiods']
        else:
            nperiods = 3

        signalList = []
        for c in df.columns:
            if c != 'Date':
                cBB = pyerb.bb(df[c], nperiods=nperiods)
                cBB['Position'] = np.nan
                cBB['Position'][(cBB['Price'] > cBB['UPPER']) & (pyerb.S(cBB['Price']) < pyerb.S(cBB['UPPER']))] = 1
                cBB['Position'][(cBB['Price'] < cBB['LOWER']) & (pyerb.S(cBB['Price']) > pyerb.S(cBB['LOWER']))] = -1
                cBB[c] = cBB['Position']
                signalList.append(cBB[c])
        s = pd.concat(signalList, axis=1).ffill().fillna(0)
        return s

    def sign_rsi(rsi, r, ob, os, **kwargs):
        if 'fix' in kwargs:
            fix = kwargs['fix']
        else:
            fix = 5
        df = rsi.copy()
        print(rsi.copy())
        print('RSI=' + str(len(df)))
        print('RSI=' + str(len(rsi)))
        df[df > ob] = 1
        df[df < os] = -1
        df.iloc[0, 0] = -1
        print(df)
        for i in range(1, len(rsi)):
            if df.iloc[i, 0] != 1 and df.iloc[i, 0] != -1:
                df.iloc[i, 0] = df.iloc[i - 1, 0]

        df = np.array(df)
        df = np.repeat(df, fix)  # fix-1 gives better sharpe
        df = pd.DataFrame(df)
        print('ASSET=' + str(r))
        print('RSI x ' + str(fix) + '=' + str(len(df)))
        c = r - len(df)  # pnl - rsi diff
        # c = r%len(df)
        print('----------------------DIFF=' + str(c))
        df = df.append(df.iloc[[-1] * c])
        df = df.reset_index(drop=True)
        print(df)
        return df

    # Advanced Operators for Portfolio Management and Optimization

    def ExPostOpt(pnl):
        MSharpe = pyerb.sharpe(pnl)
        switchFlag = np.array(MSharpe) < 0
        pnl.iloc[:, np.where(switchFlag)[0]] = pnl * (-1)
        out = [pnl, switchFlag]
        return out

    def StaticHedgeRatio(df, targetAsset):
        HedgeRatios = []
        for c in df.columns:
            subHedgeRatio = df[targetAsset].corr(df[c]) * (df[targetAsset].std()/df[c].std())
            HedgeRatios.append(subHedgeRatio)
        HedgeRatiosDF = pd.Series(HedgeRatios, index=df.columns).drop(df[targetAsset].name)
        return HedgeRatiosDF

    def BetaRegression(df, X, **kwargs):
        if 'n' in kwargs:
            n = kwargs['n']
        else:
            n = 250
        RollVolList = []
        BetaList = []
        for c in df.columns:
            if X not in c:
                RollVar_c = (1/(pyerb.S(pyerb.rollStatistics(df, 'Vol', nIn=n)**2)))
                #RollVar_c[RollVar_c > 100] = 1
                RollVolList.append(RollVar_c)
                Beta_c = (df[X].rolling(n).cov(df[c])).mul(RollVar_c, axis=0).replace([np.inf, -np.inf], 0)
                Beta_c.name = c
                BetaList.append(Beta_c)
        RollVolDF = pd.concat(RollVolList, axis=1)
        BetaDF = pd.concat(BetaList, axis=1)

        return [BetaDF, RollVolDF]

    def BetaKernel(df):

        BetaMatDF = pd.DataFrame(np.cov(df.T), index=df.columns, columns=df.columns)
        for idx, row in BetaMatDF.iterrows():
            BetaMatDF.loc[idx] /= row[idx]

        return BetaMatDF

    def MultiRegressKernel(df):

        dataList = []
        for c in df.columns:
            regr = linear_model.LinearRegression()
            regr.fit(df, df[c])
            dataList.append(regr.coef_)

        dataDF = pd.DataFrame(dataList, columns=df.columns, index=df.columns)

        return dataDF

    def RV(df, **kwargs):
        if "RVspace" in kwargs:
            RVspace = kwargs["RVspace"]
        else:
            RVspace = "classic"
        if 'mode' in kwargs:
            mode = kwargs['mode']
        else:
            mode = 'Linear'
        if 'n' in kwargs:
            n = kwargs['n']
        else:
            n = 25

        if RVspace == "classic":
            cc = list(combinations(df.columns, 2))
        elif RVspace.split("_")[0] == "specificDriver":
            cc = [c for c in list(combinations(df.columns, 2)) if c[0] == RVspace.split("_")[1]]
        elif RVspace == "specificPairs":
            cc = kwargs["targetPairs"]

        if mode == 'Linear':
            df0 = pd.concat([df[c[0]].sub(df[c[1]]) for c in cc], axis=1, keys=cc)
        elif mode == 'Baskets':
            df0 = pd.concat([df[c[0]].add(df[c[1]]) for c in cc], axis=1, keys=cc)
        elif mode == 'PriceRatio':
            df0 = pd.concat([df[c[0]]/df[c[1]] for c in cc], axis=1, keys=cc)
        elif mode == 'PriceMultiply':
            df0 = pd.concat([df[c[0]] * df[c[1]] for c in cc], axis=1, keys=cc)
        elif mode == 'PriceRatio_zScore':
            lDF = []
            for c in cc:
                PrRatio = df[c[0]] / df[c[1]]
                emaPrRatio = pyerb.ema(PrRatio, nperiods=n)
                volPrRatio = pyerb.expander(PrRatio, np.std, n)
                PrZScore = (PrRatio-emaPrRatio) / volPrRatio
                lDF.append(PrZScore)
            df0 = pd.concat(lDF, axis=1, keys=cc)
        elif mode == 'HedgeRatio':
            df0 = pd.concat([df[c[0]].rolling(n).corr(df[c[1]]) * (pyerb.roller(df[c[0]], np.std, n) / pyerb.roller(df[c[1]], np.std, n)) for c in cc], axis=1, keys=cc)
        elif mode == 'HedgeRatioInverse':
            df0 = pd.concat([df[c[0]] - (pyerb.S(df[c[0]].expanding(n).corr(df[c[1]]) * (pyerb.expander(df[c[1]], np.std, n) / pyerb.expander(df[c[0]], np.std, n)), nperiods=1)
                             * df[c[1]]) for c in cc], axis=1, keys=cc)
        elif mode == 'HedgeRatioSimpleCorr':
            df0 = pd.concat([df[c[0]] - (pyerb.S(df[c[0]].expanding(n).corr(df[c[1]]), nperiods=2) * df[c[1]]) for c in cc], axis=1, keys=cc)

        df0.columns = df0.columns.map('_'.join)

        return df0.fillna(method='ffill').fillna(0)

    def RVSignalHandler(sigDF, **kwargs):
        if 'HedgeRatioDF' in kwargs:
            HedgeRatioDF = kwargs['HedgeRatioDF']
        else:
            HedgeRatioDF = pd.DataFrame(1, index=sigDF.index, columns=sigDF.columns)
        assetSignList = []
        for c in sigDF.columns:
            medSigDF = pd.DataFrame(sigDF[c])
            HedgeRatio = HedgeRatioDF[c]
            assetNames = c.split("_")
            medSigDF[assetNames[0]] = sigDF[c]
            medSigDF[assetNames[1]] = sigDF[c] * (-1) * HedgeRatio
            subSigDF = medSigDF[[assetNames[0], assetNames[1]]]
            #print(subSigDF)
            assetSignList.append(subSigDF)
        assetSignDF = pd.concat(assetSignList, axis=1)
        #print(assetSignDF)
        assetSignDFgroupped = assetSignDF.groupby(assetSignDF.columns, axis=1).sum()
        #print(assetSignDFgroupped)
        #time.sleep(3000)
        return assetSignDFgroupped

    # Metric Build

    def Metric(metaDF, **kwargs):
        if "metric" in kwargs:
            metric = kwargs['metric']
        else:
            metric = "euclidean"
        if "minkowskiOrder" in kwargs:
            minkowskiOrder = kwargs['minkowskiOrder']
        else:
            minkowskiOrder = 3
        if "wminkowskiWeight" in kwargs:
            wminkowskiWeight = kwargs['wminkowskiWeight']
        else:
            wminkowskiWeight = 0.25
        if "seuclideanV" in kwargs:
            seuclideanV = kwargs['seuclideanV']
        else:
            seuclideanV = 1

        MetricMat = pd.DataFrame(index=metaDF.columns, columns=metaDF.columns)

        for c1 in metaDF.columns:
            for c2 in metaDF.columns:
                if metric == "euclidean":
                    MetricMat.loc[c1,c2] = np.sqrt(((metaDF[c1] - metaDF[c2])**2).sum())
                elif metric == "manhattan":
                    MetricMat.loc[c1, c2] = (metaDF[c1] - metaDF[c2]).abs().sum()
                elif metric == "chebyshev":
                    MetricMat.loc[c1, c2] = (metaDF[c1] - metaDF[c2]).abs().max()
                elif metric == "minkowski":
                    MetricMat.loc[c1, c2] = ((((metaDF[c1] - metaDF[c2]).abs())**minkowskiOrder).sum())**(1/minkowskiOrder)
                elif metric == "wminkowski":
                    MetricMat.loc[c1, c2] = ((((metaDF[c1] - metaDF[c2]) * wminkowskiWeight)**minkowskiOrder).sum())**(1/minkowskiOrder)
                elif metric == "seuclidean":
                    MetricMat.loc[c1, c2] = np.sqrt(((metaDF[c1] - metaDF[c2])**2 / seuclideanV).sum())

        eigVals, eigVecs = np.linalg.eig(MetricMat.apply(pd.to_numeric, errors='coerce').fillna(0))

        return [eigVals, eigVecs]

    # Folders Body & Plots

    def Navigate(DB, module):

        navigatorData = []
        for a in dir(module):
            if isinstance(getattr(module, a), types.FunctionType):
                # print(inspect.getfullargspec(getattr(module, a)))
                navigatorData.append(['ParentFunction', a, ','.join(getattr(module, a).__code__.co_varnames)])
            elif isinstance(getattr(module, a), types.ModuleType):
                subModule = getattr(module, a)
                for b in dir(subModule):
                    if isinstance(getattr(subModule, b), types.FunctionType):
                        navigatorData.append(
                            ['SubModuleFunction', b, ','.join(getattr(subModule, b).__code__.co_varnames)])

        navigatorDataDF = pd.DataFrame(navigatorData, columns=['FunctionType', 'FunctionName', 'Parameters'])
        navigatorDataDF.to_sql("ModuleNavigator", sqlite3.connect(DB), if_exists='replace')

    def Plot(df, **kwargs):
        if 'title' in kwargs:
            titleIn = kwargs['title']
        else:
            titleIn = 'PyERB Chart'

        if titleIn == 'ew_sharpe':
            df.plot(title="Strategy Sharpe Ratio = " + str(pyerb.sharpe(pyerb.E(df)).round(2)))
        elif titleIn == 'cs_ew_sharpe':
            fig, axes = plt.subplots(nrows=2, ncols=1)
            pyerb.cs(df).plot(ax=axes[0], title="Individual Contributions")
            pyerb.cs(pyerb.E(df)).plot(ax=axes[1])
        else:
            df.plot(title=titleIn)
        plt.show()

    def RefreshableFile(dfList, filename, refreshSecs, **kwargs):
        pd.options.display.float_format = '{:,}'.format
        pd.set_option('colheader_justify', 'center')

        if 'addButtons' in kwargs:
            addButtons = kwargs['addButtons']
        else:
            addButtons = None
        if 'addPlots' in kwargs:
            addPlots = kwargs['addPlots']
        else:
            addPlots = None
        if 'cssID' in kwargs:
            cssID = kwargs['cssID']
        else:
            cssID = ''
        if 'specificID' in kwargs:
            specificID = kwargs['specificID']
        else:
            specificID = None

        dfListNew = []
        for x in dfList:
            dfListNew.append(x[0].to_html(index=False, table_id=x[1]) + "\n\n" + "<br>")

        with open(filename, 'w') as _file:
            _file.write(''.join(dfListNew))

        append_copy = open(filename, "r")
        original_text = append_copy.read()
        append_copy.close()

        append_copy = open(filename, "w")
        if specificID not in ["DeskPnLSinceInception", "BOFFICE_GATOS_DailyPnLDF_DERIV"]:
            append_copy.write('<meta http-equiv="refresh" content="' + str(refreshSecs) + '">\n')
        append_copy.write('<meta charset="UTF-8">')
        append_copy.write('<link rel="stylesheet" href="style/df_style.css"><link rel="stylesheet" href="https://www.w3schools.com/w3css/4/w3.css"><link rel="stylesheet" href="https://fonts.googleapis.com/css?family=Inconsolata">')
        append_copy.write('<div id="footerText"> LAST UPDATE : ' + datetime.now().strftime("%m/%d/%Y, %H:%M:%S") + " - Derivatives and FX Trading Desk</div>")

        if addButtons == 'GreenBoxMain':
            append_copy.write('<div class="topnav">')
            append_copy.write(
                '<a href="file:///F:/Dealing/Panagiotis%20Papaioannou/pyerb/GreenBox/GreenBoxHome.html">GreenBox Home</a>')
            append_copy.write(
                '<a href="file:///F:/Dealing/Panagiotis Papaioannou/IRS_Risk_Reporting_Management/ALL_IRS_Risk.html">IRS Risk Reporting</a>')
            append_copy.write(
                '<a href="file:///F:/Dealing/Panagiotis%20Papaioannou/pyerb/GreenBox/QuantitativeStrategies_ControlPanel.html">Quant Strategies</a>')
            append_copy.write('</div>')
        elif 'DeskPnLSinceInception' in addButtons:
            progress = addButtons.split("_")[1]
            append_copy.write('<div class="topnav">')
            append_copy.write(
                '<a href="file:///F:/Dealing/Panagiotis%20Papaioannou/pyerb/GreenBox/GreenBoxHome.html">GreenBox Home</a>')
            append_copy.write(
                '<a href="file:///F:/Dealing/Panagiotis Papaioannou/IRS_Risk_Reporting_Management/ALL_IRS_Risk.html">IRS Risk Reporting</a>')
            append_copy.write(
                '<a href="file:///F:/Dealing/Panagiotis%20Papaioannou/pyerb/GreenBox/QuantitativeStrategies_ControlPanel.html">Quant Strategies</a>')
            append_copy.write('</div>')
            append_copy.write('<div class="topnav">')
            append_copy.write(
                '<a href="file:///F:/Dealing/Panagiotis Papaioannou/pyerb/GreenBox/SDExcelFXPricer_FX_2_Options.html">SD Pricer (FX_2)</a>')
            append_copy.write(
                '<a href="file:///F:/Dealing/Panagiotis Papaioannou/pyerb/GreenBox/BOFFICE_GATOS_DailyPnLDF_DERIV.html">BOFFICE PnL</a>')
            append_copy.write('</div>')
            append_copy.write(
                '<div class="progress"> <div class="progress-bar progress-bar-success progress-bar-striped" role="progressbar" aria-valuenow="100" aria-valuemin="0" aria-valuemax="100" style="width:'+str(progress)+'%"> '+str(progress)+'% (of EUR 5M) </div></div>')
        elif addButtons == "QuantStrategies":
            append_copy.write('<br><div class="topnav">')
            append_copy.write(
                '<a href="file:///F:/Dealing/Panagiotis%20Papaioannou/pyerb/GreenBox/GreenBoxHome.html">GreenBox Home</a>')
            append_copy.write(
                '<a href="file:///F:/Dealing/Panagiotis Papaioannou/IRS_Risk_Reporting_Management/ALL_IRS_Risk.html">IRS Risk Reporting</a>')
            append_copy.write(
                '<a href="file:///F:/Dealing/Panagiotis%20Papaioannou/pyerb/GreenBox/QuantitativeStrategies_ControlPanel.html">Quant Strategies</a>')
            append_copy.write(
                '<a href="file:///F:/Dealing/Panagiotis Papaioannou/pyerb/GreenBox/PPAPAIOANNO1_trader_aggregatedFills.html">EMSX Expiries</a>')
            append_copy.write(
                '<a href="file:///F:/Dealing/Panagiotis%20Papaioannou/pyerb/GreenBox/DeskPnLSinceInception.html">Desk PnL</a>')
            append_copy.write('</div>')
            append_copy.write('<div class="topnav">')
            append_copy.write(
                '<a href="file:///F:/Dealing/Panagiotis Papaioannou/pyerb/PyEurobankBloomberg/PySystems/PyLiveTradingSystems/StrategiesFactSheets/Endurance.html">Endurance</a>')
            append_copy.write(
                '<a href="file:///F:/Dealing/Panagiotis Papaioannou/pyerb/PyEurobankBloomberg/PySystems/PyLiveTradingSystems/StrategiesFactSheets/Coast.html">Coast</a>')
            append_copy.write(
                '<a href="file:///F:/Dealing/Panagiotis Papaioannou/pyerb/PyEurobankBloomberg/PySystems/PyLiveTradingSystems/StrategiesFactSheets/Brotherhood.html">Brotherhood</a>')
            append_copy.write(
                '<a href="file:///F:/Dealing/Panagiotis Papaioannou/pyerb/PyEurobankBloomberg/PySystems/PyLiveTradingSystems/StrategiesFactSheets/Endurance_Coast_Brotherhood_TimeStory.html">Endurance + Coast + Brotherhood</a>')
            append_copy.write(
                '<a href="file:///F:/Dealing/Panagiotis Papaioannou/pyerb/GreenBox/BetaEyeDF_MajorFutures.html">Betas</a>')
            append_copy.write('</div><br>')
        elif addButtons == "aggregatedFills":
            append_copy.write('<div class="topnav">')
            append_copy.write(
                '<a href="file:///F:/Dealing/Panagiotis%20Papaioannou/pyerb/GreenBox/GreenBoxHome.html">GreenBox Home</a>')
            append_copy.write(
                '<a href="file:///F:/Dealing/Panagiotis Papaioannou/IRS_Risk_Reporting_Management/ALL_IRS_Risk.html">IRS Risk Reporting</a>')
            append_copy.write(
                '<a href="file:///F:/Dealing/Panagiotis%20Papaioannou/pyerb/GreenBox/QuantitativeStrategies_ControlPanel.html">Quant Strategies</a><br>')
            append_copy.write('</div>')

            append_copy.write('<div class="topnav">')
            ### STRATEGIES PAGES ###
            append_copy.write(
                '<a href="file:///F:/Dealing/Panagiotis Papaioannou/pyerb/GreenBox/CKATSILEROS1_trader_aggregatedFills.html">C. Katsileros</a>')
            append_copy.write(
                '<a href="file:///F:/Dealing/Panagiotis Papaioannou/pyerb/GreenBox/NASIMAKOPOU1_trader_aggregatedFills.html">N. Assimakopoulos</a>')
            append_copy.write(
                '<a href="file:///F:/Dealing/Panagiotis Papaioannou/pyerb/GreenBox/SDALIAKOPOUL_trader_aggregatedFills.html">S. Daliakopoulos</a>')
            append_copy.write(
                '<a href="file:///F:/Dealing/Panagiotis Papaioannou/pyerb/GreenBox/PPAPAIOANNO1_trader_aggregatedFills.html">P. Papaioannou</a>')

            append_copy.write('</div>')

        if addPlots is not None:
            append_copy.write('<p>')
            append_copy.write('<img src="'+addPlots+'" alt="'+addPlots+'Img" width="500" height="400">')
            append_copy.write('</p>')

        if specificID == "DeskPnLSinceInception":
            append_copy.write(original_text)
            append_copy.write(
                '<br><div id="chartDiv" style="width:80%; height:650px; margin:0 auto;"></div><br>'
                "<script src='http://code.jquery.com/jquery-1.9.1.js'></script>"
                '<script src="https://code.jscharting.com/2.9.0/jscharting.js"></script>'
                '<script src="js/rsPyDeskPnL.js"></script>'
            )
            append_copy.write(
                "<script>  $(function() {$('table td').each(function(){var txt = $(this).text(); if(txt.includes('Endurance')) $(this).css('color', '#2fd10f'); else if(txt.includes('Coast')) $(this).css('color', '#21ebd3'); if(parseInt(txt) < 0 && !txt.includes('M') && !txt.includes('Y'))  $(this).css('color', 'red'); else if(parseInt(txt) > 0 && !txt.includes('M') && !txt.includes('Y')) $(this).css('color', '#1193fa');});});</script><script>$('table').each (function(){var tableElementID = $(this).closest('table').attr('id'); $('#'+tableElementID+' thead>tr>th').each (function(index){var txt = $(this).text(); if(txt.includes('Asset')||txt.includes('TOTAL')) {var SumCol = index; $('#'+tableElementID+' tr').each(function() { $(this).find('td').each (function(index) {if(index === SumCol) {$(this).css('background', 'black'); $(this).css('color', ' #f2a20d'); $(this).css('border', '2px solid white');}})})}});});</script>")
            append_copy.close()
        elif specificID == "BOFFICE_GATOS_DailyPnLDF_DERIV":
            append_copy.write(original_text)
            append_copy.write(
                '<br><div id="chartDiv" style="width:80%; height:650px; margin:0 auto;"></div><br>'
                "<script src='http://code.jquery.com/jquery-1.9.1.js'></script>"
                '<script src="https://code.jscharting.com/2.9.0/jscharting.js"></script>'
                '<script src="js/BOFFICE_GATOS_DailyPnLDF_DERIV.js"></script>'
            )
            append_copy.write(
                "<script>  $(function() {$('table td').each(function(){var txt = $(this).text(); if(txt.includes('Endurance')) $(this).css('color', '#2fd10f'); else if(txt.includes('Coast')) $(this).css('color', '#21ebd3'); if(parseInt(txt) < 0 && !txt.includes('M') && !txt.includes('Y'))  $(this).css('color', 'red'); else if(parseInt(txt) > 0 && !txt.includes('M') && !txt.includes('Y')) $(this).css('color', '#1193fa');});});</script><script>$('table').each (function(){var tableElementID = $(this).closest('table').attr('id'); $('#'+tableElementID+' thead>tr>th').each (function(index){var txt = $(this).text(); if(txt.includes('Asset')||txt.includes('TOTAL')) {var SumCol = index; $('#'+tableElementID+' tr').each(function() { $(this).find('td').each (function(index) {if(index === SumCol) {$(this).css('background', 'black'); $(this).css('color', ' #f2a20d'); $(this).css('border', '2px solid white');}})})}});});</script>")
            append_copy.close()
        elif specificID == 'BetaEyeDF':
            append_copy.write(
                '<a href="file:///F:/Dealing/Panagiotis%20Papaioannou/pyerb/GreenBox/Beta_RX1%20Comdty.html"  id="BetaBundTS">(RX1 Comdty) Bund Betas</a>&nbsp &nbsp &nbsp')
            append_copy.write(
                '<a href="file:///F:/Dealing/Panagiotis%20Papaioannou/pyerb/GreenBox/Beta_Endurance.html"  id="BetaEnduranceTS">(Macros) Endurance Betas</a>&nbsp &nbsp &nbsp')
            append_copy.write('<br>')
            append_copy.write(original_text)
            append_copy.write(
                "<script src='http://code.jquery.com/jquery-1.9.1.js'></script><script>  $(function() {$('table td').each(function(){var txt = $(this).text(); "
                "if(parseInt(txt) < 0)  $(this).css('color', 'red');  else if(parseInt(txt) > 0) $(this).css('color', '#1193fa');});});</script>")
            append_copy.write(
                "<script>$('table').each (function(){var tableElementID = $(this).closest('table').attr('id'); $('#'+tableElementID+' thead>tr>th').each (function(index){var txt = $(this).text(); if(txt.includes('index')) {var SumCol = index; $('#'+tableElementID+' tr').each(function() { $(this).find('td').each (function(index) {if(index === SumCol) {$(this).css('color', 'white');}})})}});});</script>")
            append_copy.close()
        else:
            append_copy.write(original_text)
            append_copy.write(
                "<script src='http://code.jquery.com/jquery-1.9.1.js'></script><script>  $(function() {$('table td').each(function(){var txt = $(this).text(); "
                "if(txt.includes('Endurance')) $(this).css('color', '#2fd10f'); else if(txt.includes('Coast')) $(this).css('color', '#21ebd3'); "
                "if(parseInt(txt) < 0 && !txt.includes('M') && !txt.includes('Y'))  $(this).css('color', 'red'); else if (txt.includes('NEED TO ROLL !!!'))  $(this).css('color', 'green'); else if (txt.includes('Expired'))  $(this).css('color', 'white'); else if(parseInt(txt) > 0 && !txt.includes('M') && !txt.includes('Y')) $(this).css('color', '#1193fa');});});</script>")
            append_copy.write(
                "<script>$('table').each (function(){var tableElementID = $(this).closest('table').attr('id'); $('#'+tableElementID+' thead>tr>th').each (function(index){var txt = $(this).text(); if(txt.includes('Asset')||txt.includes('TOTAL')) {var SumCol = index; $('#'+tableElementID+' tr').each(function() { $(this).find('td').each (function(index) {if(index === SumCol) {$(this).css('background', 'black'); $(this).css('color', ' #f2a20d'); $(this).css('border', '2px solid white');}})})}});});</script>")
            append_copy.close()

# SubClasses
class BackTester:

    def backTestReturnKernel(kernel, tradedAssets, **kwargs):

        if 'mode' in kwargs:
            mode = kwargs['mode']
        else:
            mode = 'directionalPredictability'

        if 'TrShift' in kwargs:
            TrShift = kwargs['TrShift']
        else:
            TrShift = 1

        if 'reverseFlag' in kwargs:
            reverseFlag = kwargs['reverseFlag']
        else:
            reverseFlag = 1

        if 'scanAll' in kwargs:
            scanAll = kwargs['scanAll']
        else:
            scanAll = 'no'

        if mode == 'directionalPredictability':
            kernel = pyerb.sign(kernel)
        else:
            print("Using defaulet 'Direct trading kernel projection' to traded assets ...")

        if isinstance(kernel, pd.Series):
            kernel = pd.DataFrame(kernel)
        if isinstance(tradedAssets, pd.Series):
            tradedAssets = pd.DataFrame(tradedAssets)

        if (len(kernel.columns) != len(tradedAssets.columns)) | (scanAll == 'yes'):
            print("Kernel's dimension is not the same with the dimension of the Traded Assets matrix - building BT crosses...")
            cc = []
            for ck in kernel.columns:
                for c in tradedAssets.columns:
                    cc.append((ck, c))
            pnl = pd.concat([pyerb.S(kernel[c[0]], nperiods=TrShift) * pyerb.dlog(tradedAssets[c[1]]) * reverseFlag for c in cc], axis=1, keys=cc)
        else:
            print("Straight BT Projection...")
            pnl = pyerb.S(kernel, nperiods=TrShift) * pyerb.dlog(tradedAssets) * reverseFlag
        return pnl

class AI:

    def gRollingManifold(manifoldIn, df0, st, NumProjections, eigsPC, **kwargs):
        if 'RollMode' in kwargs:
            RollMode = kwargs['RollMode']
        else:
            RollMode = 'RollWindow'

        if 'Scaler' in kwargs:
            Scaler = kwargs['Scaler']
        else:
            Scaler = 'Standard'

        if 'ProjectionMode' in kwargs:
            ProjectionMode = kwargs['ProjectionMode']
        else:
            ProjectionMode = 'NoTranspose'

        if 'contractiveObserver' in kwargs:
            contractiveObserver = kwargs['contractiveObserver']
        else:
            contractiveObserver = 0

        if 'LLE_n_neighbors' in kwargs:
            n_neighbors = kwargs['LLE_n_neighbors']
        else:
            n_neighbors = 2

        if 'LLE_Method' in kwargs:
            LLE_Method = kwargs['LLE_n_neighbors']
        else:
            LLE_Method = 'standard'

        if 'DMAPS_sigma' in kwargs:
            sigma = kwargs['DMAPS_sigma']
        else:
            sigma = 'std'

        "CALCULATE ROLLING STATISTIC"
        if manifoldIn == 'CustomMetric':
            if 'CustomMetricStatistic' in kwargs:
                CustomMetricStatistic = kwargs['CustomMetricStatistic']
                metaDF_Rolling = pyerb.rollStatistics(df0.copy(), CustomMetricStatistic)
            else:
                CustomMetricStatistic = None
                metaDF_Rolling = df0.copy()

            if 'CustomMetric' in kwargs:
                CustomMetric = kwargs['CustomMetric']
            else:
                CustomMetric = "euclidean"

        eigDf = []
        eigCoeffs = [[] for j in range(len(eigsPC))]
        Comps = [[] for j in range(len(eigsPC))]
        sigmaList = []
        lambdasList = []
        cObserverList = []
        # st = 50; pcaN = 5; eigsPC = [0];
        for i in tqdm(range(st, len(df0) + 1)):
            # try:

            #print("Step:", i, " of ", len(df0) + 1)
            if RollMode == 'RollWindow':
                df = df0.iloc[i - st:i, :]
            else:
                df = df0.iloc[0:i, :]

            if ProjectionMode == 'Transpose':
                df = df.T

            features = df.columns.values
            x = df.loc[:, features].values

            if Scaler == 'Standard':
                x = StandardScaler().fit_transform(x)

            if manifoldIn == 'CustomMetric':

                customMetric = pyerb.Metric(metaDF_Rolling, statistic=CustomMetricStatistic, metric=CustomMetric)
                lambdasList.append(list(customMetric[0]))
                sigmaList.append(list(customMetric[0]))
                c = 0
                for eig in eigsPC:
                    #print(eig, ', customMetric[1][eig] =', customMetric[1][eig]) # 0 , 100 , 5
                    Comps[c].append(list(customMetric[1][eig]))
                    c += 1

            elif manifoldIn == 'PCA':
                pca = PCA(n_components=NumProjections)
                X_pca = pca.fit_transform(x)
                lambdasList.append(list(pca.singular_values_))
                sigmaList.append(list(pca.explained_variance_ratio_))
                c = 0
                for eig in eigsPC:
                    #print(eig, ',', len(pca.components_[eig]), ',', len(pca.components_)) # 0 , 100 , 5
                    Comps[c].append(list(pca.components_[eig]))
                    c += 1

            elif manifoldIn == 'BetaRegressV':
                BetaKernelDF = pyerb.BetaKernel(df)

                lambdasList.append(1)
                sigmaList.append(1)
                c = 0
                for eig in eigsPC:
                    Comps[c].append(BetaKernelDF.iloc[:,eig].tolist())
                    c += 1

            elif manifoldIn == 'BetaProject':
                BetaKernelDF = pyerb.BetaKernel(df)

                lambdasList.append(1)
                sigmaList.append(1)
                c = 0
                for eig in eigsPC:
                    Comps[c].append(BetaKernelDF.iloc[:,eig].tolist())
                    c += 1

            elif manifoldIn == 'BetaRegressH':
                BetaKernelDF = pyerb.BetaKernel(df)

                lambdasList.append(1)
                sigmaList.append(1)
                c = 0
                for eig in eigsPC:
                    Comps[c].append(BetaKernelDF.iloc[eig,:].tolist())
                    c += 1

            elif manifoldIn == 'BetaRegressC':
                BetaKernelDF = pyerb.BetaKernel(df)

                lambdasList.append(1)
                sigmaList.append(1)
                c = 0
                for eig in eigsPC:
                    Comps[c].append((BetaKernelDF.iloc[eig,:]+BetaKernelDF.iloc[:,eig]).tolist())
                    c += 1

            elif manifoldIn == 'BetaDiffusion':
                BetaKernelDF = pyerb.BetaKernel(df)
                BetaKernelDF *= 1/BetaKernelDF.median()

                U, s, VT = svd(BetaKernelDF.values)

                lambdasList.append(s)
                sigmaList.append(1)
                c = 0
                for eig in eigsPC:
                    Comps[c].append(U[eig])
                    c += 1

            elif manifoldIn == 'Beta':
                BetaKernelDF = pyerb.BetaKernel(df)
                U, s, VT = svd(BetaKernelDF.values)

                lambdasList.append(s)
                sigmaList.append(1)
                c = 0
                for eig in eigsPC:
                    Comps[c].append(U[eig])
                    c += 1

            elif manifoldIn == 'MultipleRegress':

                MRKernelDF = pyerb.MultiRegressKernel(df)

                lambdasList.append(1)
                sigmaList.append(1)
                c = 0
                for eig in eigsPC:
                    Comps[c].append(MRKernelDF.iloc[eig, :].tolist())
                    c += 1

            elif manifoldIn == 'DMAPS':
                dMapsOut = pyerb.AI.gDmaps(df, nD=NumProjections, coFlag=contractiveObserver,
                                            sigma=sigma)  # [eigOut, sigmaDMAPS, s[:nD], glA]
                eigDf.append(dMapsOut[0].iloc[-1, :])
                glAout = dMapsOut[3]
                cObserverList.append(dMapsOut[4].iloc[-1, :])
                sigmaList.append(dMapsOut[1])
                lambdasList.append(dMapsOut[2])
                for gl in glAout:
                    Comps[gl].append(glAout[gl])
                    eigCoeffs[gl].append(
                        linear_model.LinearRegression(normalize=True).fit(df, dMapsOut[0].iloc[:, gl]).coef_)

            elif manifoldIn == 'LLE':
                lle = manifold.LocallyLinearEmbedding(n_neighbors=n_neighbors, n_components=NumProjections,
                                                      method=LLE_Method, n_jobs=-1)
                X_lle = lle.fit_transform(x)  # ; print(X_lle.shape)
                lambdasList.append(1)
                sigmaList.append(1)
                c = 0
                for eig in eigsPC:
                    # print(eig, ',', len(X_lle[:, eig])) # 0 , 100 , 5
                    Comps[c].append(list(X_lle[:, eig]))
                    c += 1

            # except Exception as e:
            #    print(e)
            #    for c in len(eigsPC):
            #        Comps[c].append(list(np.zeros(len(df0.columns), 1)))
            #        eigCoeffs[c].append(list(np.zeros(len(df0.columns), 1)))

        sigmaDF = pd.concat([pd.DataFrame(np.zeros((st - 1, 1))), pd.DataFrame(sigmaList)], axis=0,
                            ignore_index=True).fillna(0)
        sigmaDF.index = df0.index
        try:
            if len(sigmaDF.columns) <= 1:
                sigmaDF.columns = ['sigma']
        except Exception as e:
            print(e)

        lambdasDF = pd.concat(
            [pd.DataFrame(np.zeros((st - 1, pd.DataFrame(lambdasList).shape[1]))), pd.DataFrame(lambdasList)],
            axis=0, ignore_index=True).fillna(0)
        lambdasDF.index = df0.index

        if contractiveObserver == 0:
            principalCompsDf = [[] for j in range(len(Comps))]
            exPostProjections = [[] for j in range(len(Comps))]
            for k in range(len(Comps)):
                # principalCompsDf[k] = pd.DataFrame(pcaComps[k], columns=df0.columns, index=df1.index)

                principalCompsDf[k] = pd.concat(
                    [pd.DataFrame(np.zeros((st - 1, len(df0.columns))), columns=df0.columns),
                     pd.DataFrame(Comps[k], columns=df0.columns)], axis=0, ignore_index=True)
                principalCompsDf[k].index = df0.index
                principalCompsDf[k] = principalCompsDf[k].fillna(0).replace(0, np.nan).ffill()

                exPostProjections[k] = df0 * pyerb.S(principalCompsDf[k])

            return [df0, principalCompsDf, exPostProjections, sigmaDF, lambdasDF]

        else:

            return [df0, pd.DataFrame(eigDf), pd.DataFrame(cObserverList), sigmaDF, lambdasDF, Comps, eigCoeffs]

class PyBloomberg:

    def __init__(self, DB):
        self.DB = DB

    def CustomDataFetch(TargetDB, name, assetsToProcess, fieldsIn):

        dataOutCustomData = []

        EXCEPTIONS = blpapi.Name("exceptions")
        FIELD_ID = blpapi.Name("fieldId")
        REASON = blpapi.Name("reason")
        CATEGORY = blpapi.Name("category")
        DESCRIPTION = blpapi.Name("description")
        ERROR_INFO = blpapi.Name("ErrorInfo")

        class Window(object):
            def __init__(self, name):
                self.name = name

            def displaySecurityInfo(self, msg):
                print("%s: %s" % (self.name, msg))
                # print("%s:" % (self.name))

                d = msg.getElement('securityData')
                size = d.numValues()
                fieldDataList = [[d.getValueAsElement(i).getElement("security").getValueAsString(),
                                  d.getValueAsElement(i).getElement("fieldData")] for i in range(0, size)]
                for x in fieldDataList:
                    subData = []
                    # print(x, "^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
                    for fld in fieldsIn:
                        try:
                            subData.append([fld, x[1].getElement(fld).getValueAsString()])
                        except Exception as e:
                            # print(e)
                            subData.append([fld, np.nan])
                        subDataDF = pd.DataFrame(subData, columns=["field", x[0]]).set_index('field', drop=True)
                    dataOutCustomData.append(subDataDF)

                dfOUT = pd.concat(dataOutCustomData, axis=1).T
                dfOUT.to_sql(name, sqlite3.connect(TargetDB), if_exists='replace')

        def parseCmdLine():
            parser = OptionParser(description="Retrieve reference data.")
            parser.add_option("-a",
                              "--ip",
                              dest="host",
                              help="server name or IP (default: %default)",
                              metavar="ipAddress",
                              default="localhost")
            parser.add_option("-p",
                              dest="port",
                              type="int",
                              help="server port (default: %default)",
                              metavar="tcpPort",
                              default=8194)

            (options, args) = parser.parse_args()

            return options

        def startSession(session):
            if not session.start():
                print("Failed to connect!")
                return False

            if not session.openService("//blp/refdata"):
                print("Failed to open //blp/refdata")
                session.stop()
                return False

            return True

        global options
        options = parseCmdLine()

        # Fill SessionOptions
        sessionOptions = blpapi.SessionOptions()
        sessionOptions.setServerHost(options.host)
        sessionOptions.setServerPort(options.port)

        print("Connecting to %s:%d" % (options.host, options.port))

        # Create a Session
        session = blpapi.Session(sessionOptions)

        # Start a Session
        if not startSession(session):
            return

        refDataService = session.getService("//blp/refdata")
        request = refDataService.createRequest("ReferenceDataRequest")
        for asset in assetsToProcess:
            request.append("securities", asset)
        for fld in fieldsIn:
            request.append("fields", fld)

        secInfoWindow = Window("SecurityInfo")
        cid = blpapi.CorrelationId(secInfoWindow)

        # print("Sending Request:", request)
        session.sendRequest(request, correlationId=cid)

        try:
            # Process received events
            while (True):
                # We provide timeout to give the chance to Ctrl+C handling:
                event = session.nextEvent(500)
                for msg in event:
                    if event.eventType() == blpapi.Event.RESPONSE or \
                            event.eventType() == blpapi.Event.PARTIAL_RESPONSE:
                        window = msg.correlationIds()[0].value()
                        window.displaySecurityInfo(msg)

                # Response completly received, so we could exit
                if event.eventType() == blpapi.Event.RESPONSE:
                    break
        finally:
            # Stop the session
            session.stop()
