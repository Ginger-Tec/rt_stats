# metrics.pyx

import numpy as np
cimport numpy as np

cdef double calculate_return_over_period(np.ndarray[double, ndim=1] returns_array_reverse, int days):
    cdef int len_array = len(returns_array_reverse)
    if len_array > days:
        return returns_array_reverse[days]
    return np.nan

cdef double calculate_omega_ratio(np.ndarray[double, ndim=1] returns_array, double threshold=0):
    cdef np.ndarray[double, ndim=1] excess_returns = returns_array - threshold
    cdef double positive_returns = excess_returns[excess_returns > 0].sum()
    cdef double negative_returns = -excess_returns[excess_returns < 0].sum()

    if negative_returns == 0:
        return np.nan if positive_returns == 0 else np.inf

    return positive_returns / negative_returns

cdef double calculate_var(np.ndarray[double, ndim=1] returns_array, double confidence_level=0.95):
    return np.percentile(returns_array, (1 - confidence_level) * 100)

cdef np.ndarray[double, ndim=1] compute_cum_max(np.ndarray[double, ndim=1] arr):
    cdef int len_arr = len(arr)
    cdef np.ndarray[double, ndim=1] cum_max = np.empty_like(arr)
    cdef double current_max = arr[0]
    cdef int i
    for i in range(len_arr):
        if arr[i] > current_max:
            current_max = arr[i]
        cum_max[i] = current_max
    return cum_max

cdef np.ndarray[double, ndim=1] rolling_window_mean(np.ndarray[double, ndim=1] arr, int window):
    cdef int len_result = len(arr) - window + 1
    cdef np.ndarray[double, ndim=1] result = np.empty(len_result)
    cdef int i
    for i in range(len_result):
        result[i] = arr[i:i + window].mean()
    return result

cdef np.ndarray[double, ndim=1] rolling_window_std(np.ndarray[double, ndim=1] arr, int window):
    cdef int len_result = len(arr) - window + 1
    cdef np.ndarray[double, ndim=1] result = np.empty(len_result)
    cdef int i
    for i in range(len_result):
        result[i] = arr[i:i + window].std()
    return result

def metrics(np.ndarray[double, ndim=1] returns_array, bint full=False):
    cdef int total_days = len(returns_array)
    cdef double cumulative_return, cagr, mdd, mean_return, volatility, sharpe_ratio
    cdef double mean_win_return, mean_loss_return, rolling_mean_20d, rolling_std_20d
    cdef double recovery_factor, ulcer_index, serenity_index, calmar_ratio, skewness, kurtosis
    cdef double omega_ratio, daily_var
    cdef int win_days, loss_days, longest_dd_days
    cdef int drawdown_duration, max_drawdown_duration
    cdef double val

    # 최근 기간 수익률 처리
    cdef np.ndarray[double, ndim=1] returns_array_reverse = np.cumprod(1 + returns_array[::-1]) - 1

    # 누적 수익률
    cumulative_return = returns_array_reverse[-1]

    # 특정 기간 수익률 계산
    periods = [20, 40, 60, 120, 240]  # Python 리스트로 초기화
    returns_over_periods = {f'{d}d': calculate_return_over_period(returns_array_reverse, d) for d in periods}

    # CAGR (연평균 수익률)
    years = total_days / 252
    cagr = (cumulative_return + 1) ** (1 / years) - 1

    # MDD (최대 낙폭)
    cdef np.ndarray[double, ndim=1] cum_returns = np.cumprod(1 + returns_array)
    cdef np.ndarray[double, ndim=1] peak = compute_cum_max(cum_returns)
    cdef np.ndarray[double, ndim=1] drawdown = (cum_returns - peak) / peak
    mdd = drawdown.min()

    # 평균 수익률
    mean_return = returns_array.mean()

    # 변동성
    volatility = returns_array.std() * np.sqrt(252)

    # 샤프 비율 (무위험 수익률은 0으로 가정)
    sharpe_ratio = mean_return / volatility * np.sqrt(252)

    # 승리한 날과 패배한 날의 수 계산
    win_days = np.sum(returns_array > 0)
    loss_days = np.sum(returns_array < 0)

    # 수익 평균 수익률과 손실 평균 수익률 계산
    mean_win_return = np.mean(returns_array[returns_array > 0])
    mean_loss_return = np.mean(returns_array[returns_array < 0])

    # 기본적인 통계량 딕셔너리 생성
    cdef dict stats = {
        'Cumulative': cumulative_return,
        'CAGR': cagr,
        'MDD': mdd,
        'Mean': mean_return,
        'Volatility': volatility,
        'Sharpe': sharpe_ratio,
        'Mean Win Return': mean_win_return,
        'Mean Loss Return': mean_loss_return,
        'Total Days': total_days,
        'Win Days': win_days,
        'Loss Days': loss_days,
    }

    # full 옵션이 True일 때만 추가 통계량을 계산
    if full:
        # 20일 롤링 수익률의 평균 및 표준편차 계산
        rolling_mean_20d = rolling_window_mean(returns_array, 20).mean()
        rolling_std_20d = rolling_window_std(returns_array, 20).mean()

        # Recovery Factor
        recovery_factor = cumulative_return / abs(mdd) if mdd != 0 else np.nan

        # Ulcer Index
        ulcer_index = np.sqrt(np.mean(drawdown ** 2))

        # Serenity Index
        serenity_index = cagr / (ulcer_index * volatility) if ulcer_index != 0 else np.nan

        # Longest DD Days
        drawdown_duration = 0
        max_drawdown_duration = 0
        for val in drawdown:
            if val < 0:
                drawdown_duration += 1
                if drawdown_duration > max_drawdown_duration:
                    max_drawdown_duration = drawdown_duration
            else:
                drawdown_duration = 0
        longest_dd_days = max_drawdown_duration

        # Calmar Ratio
        calmar_ratio = cagr / abs(mdd) if mdd != 0 else np.nan

        # Skewness
        skewness = ((returns_array - mean_return) ** 3).mean() / (returns_array.std() ** 3)

        # Kurtosis
        kurtosis = ((returns_array - mean_return) ** 4).mean() / (returns_array.std() ** 4) - 3

        # Omega Ratio
        omega_ratio = calculate_omega_ratio(returns_array)

        # Daily VaR
        daily_var = calculate_var(returns_array)

        # 추가 통계량을 stats 딕셔너리에 추가
        stats.update({
            'Ulcer Index': ulcer_index,
            'Daily VaR': daily_var,
            'Skew': skewness,
            'Kurtosis': kurtosis,
            'Longest DD Days': longest_dd_days,
            'Recovery Factor': recovery_factor,
            'Serenity Index': serenity_index,
            'Calmar': calmar_ratio,
            'Omega': omega_ratio,
            'Rolling Mean 20d': rolling_mean_20d,
            'Rolling Std 20d': rolling_std_20d,
        })

    # 특정 기간 수익률 추가
    stats.update(returns_over_periods)

    return stats