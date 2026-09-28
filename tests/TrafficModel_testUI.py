import os as _os
_os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
_os.environ.setdefault('OMP_NUM_THREADS', '1')
_os.environ.setdefault('MKL_NUM_THREADS', '1')
_os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')

import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd
from multiprocessing import Pool, cpu_count, freeze_support
import time
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import os
import threading
import gc
import copy
import itertools

# Судопроизводство: используется обученная ML-модель (court_decision_baseline)
from court_decision_baseline import DecisionModelSystem
from legal_regulation import (enrich_regulatory_features, constrain_prediction_arrays,
                              LEGAL_OVERRIDE_DEFAULTS, LEGAL_OVERRIDE_MAP)

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


def resolve_project_path(path):
    if not path:
        return path
    path = os.path.expanduser(str(path))
    return path if os.path.isabs(path) else os.path.join(PROJECT_ROOT, path)


def build_legal_overrides(params, gov_type='lite'):
    """Переопределения санкций для типа правительства: только параметры, отличные от дефолтов.

    Ключи результата — базовые имена параметров (без суффикса _l/_s), значения — числа.
    Базой сравнения служат мягкие (lite) значения по умолчанию: строгие штрафы
    (max_fee_12.8_1, min_fee_264.1, max_fee_264.1) всегда передаются явно.
    При значениях по умолчанию для мягкого типа возвращает пустой словарь.
    """
    suffix = '_l' if gov_type == 'lite' else '_s'
    overrides = {}
    for mapping in LEGAL_OVERRIDE_MAP.values():
        for base in mapping:
            key = base + suffix
            default = LEGAL_OVERRIDE_DEFAULTS[base + '_l']
            value = params.get(key, default)
            if not np.isclose(float(value), float(default)):
                overrides[base] = float(value)
    return overrides


def build_court_pool(court, params):
    """Два пула вердиктов: мягкий (lite) и строгий (strict), с переопределениями."""
    ovr_lite = build_legal_overrides(params, 'lite')
    ovr_strict = build_legal_overrides(params, 'strict')
    return {'lite': court.build_pool(overrides=ovr_lite),
            'strict': court.build_pool(overrides=ovr_strict)}


class Driver:
    # Массивы вместо словарей: меньше памяти, выше скорость.
    __slots__ = ['pay', 'wait', 'speeding', 'mu', 'sigma', 'exp',
                 'length', 'speed', 'belt', 'crashed', 'checked', 'fataled',
                 'drunk_driving_chance', 'drunk', 'court',
                 'd_imprisoment_epoch', 'd_license_revocation_epoch', 'd_condition',
                 'd_fine_rub', 'd_reconcilation', 'd_costs_on_state', 'd_costs_amount_rub',
                 'd_travel_ban', 'd_self_recusal']

    def __init__(self, speeding, speed, exp_road_duration, sigma_road_duration, length, belt,
                 drunk_driving_chance=0.5):
        self.pay = 0
        self.wait = 0
        self.speeding = bool(speeding)
        self.mu = exp_road_duration
        self.sigma = sigma_road_duration
        self.exp = np.random.normal(self.mu, self.sigma)
        self.length = length
        self.speed = speed
        self.belt = belt
        self.crashed = False
        self.checked = False
        self.fataled = False
        # Пьянство и суд
        self.drunk_driving_chance = drunk_driving_chance   # шанс пьянства
        self.drunk = False
        self.court = False
        self.d_imprisoment_epoch = 0
        self.d_license_revocation_epoch = 0
        self.d_condition = 0
        self.d_fine_rub = 0.0
        self.d_reconcilation = 0
        self.d_costs_on_state = 0
        self.d_costs_amount_rub = 0.0
        self.d_travel_ban = 0
        self.d_self_recusal = 0

    def rerun_exp(self):
        self.exp = int(np.random.normal(self.mu, self.sigma))

    def law(self):
        if self.speeding:
            print("Fast")
        else:
            print("Norm")


class Goverment:
    __slots__ = ['fatal', 'spend', 'pay', 'isStrict', 'crash_chance', 'crash',
                 'mu', 'sigma', 'wait', 'duration', 'check_chance', 'spendings',
                 'fee_belt', 'fee_speeding', 'maintance_coeff',
                 'feeD', 'feeLeg']

    def __init__(self, isStrict, car_crash_chance, mu_wait, sigma_wait,
                 check_duration, check_chance, traffic_spendings,
                 fee_belt, fee_speeding, maintance_coeff, fatal):

        self.fatal = fatal
        self.spend = 0
        self.pay = 0
        self.isStrict = bool(isStrict)
        self.crash_chance = car_crash_chance
        self.crash = bool(np.random.binomial(1, car_crash_chance))
        self.mu = mu_wait
        self.sigma = sigma_wait
        self.wait = np.random.normal(self.mu, self.sigma)
        self.duration = check_duration
        self.check_chance = check_chance
        self.spendings = traffic_spendings
        self.fee_belt = fee_belt
        self.fee_speeding = fee_speeding
        self.maintance_coeff = maintance_coeff
        # Суд
        self.feeD = 0.0
        self.feeLeg = 0.0

    def rerun_wait(self):
        self.wait = np.random.normal(self.mu, self.sigma)

    def gov(self):
        if self.isStrict:
            print("Strict")
        else:
            print("Lite")



ML_TO_DOC_FIELDS = {
    'd_imprisonment_epochs': 'd_imprisoment_epoch',             
    'd_license_revocation_epochs': 'd_license_revocation_epoch', 
    'd_conditional': 'd_condition',                              
    'd_reconciliation': 'd_reconcilation',                       
}


def remap_verdict_fields(verdict):
    return {ML_TO_DOC_FIELDS.get(key, key): value for key, value in verdict.items()}


def _register_main_model_classes():
    import sys
    import court_decision_baseline as cdb
    main = sys.modules.get('__main__')
    if main is None:
        return
    for name, obj in vars(cdb).items():
        if isinstance(obj, type) and not hasattr(main, name):
            setattr(main, name, obj)


class CourtSystem:
    """Система судопроизводства: пул вердиктов из обученной ML-модели (строится один раз, передаётся в процессы)."""

    def __init__(self, checkpoint_path, dataset_path, pool_size, pool=None):
        # Пути относительно корня проекта (или абсолютные) -> абсолютные
        self.checkpoint_path = resolve_project_path(checkpoint_path)
        self.dataset_path = resolve_project_path(dataset_path)
        self.pool_size = pool_size
        self.pool = pool
        self._system = None

    def _load_system(self):
        if self._system is None:
            _register_main_model_classes()
            self._system = DecisionModelSystem.load(self.checkpoint_path)
        return self._system

    def load_case_dataset(self):
        """Чтение датасета дел; берутся только колонки c_*."""
        import pandas as pd
        df = pd.read_csv(self.dataset_path, low_memory=False)
        c_cols = [col for col in df.columns if col.startswith('c_')]
        if not c_cols:
            raise ValueError(
                "CSV {0} не содержит колонок c_* (ожидается формат vectors.csv).".format(self.dataset_path))
        return df[c_cols]

    def _cache_path(self, overrides=None):
        """Путь к кэш-файлу пула вердиктов; суффикс _ovr_<hash> только при непустых переопределениях."""
        import os
        import hashlib
        dataset_abs = os.path.abspath(self.dataset_path)
        dataset_key = hashlib.md5(dataset_abs.encode('utf-8')).hexdigest()[:16]
        cache_dir = os.path.join(os.path.dirname(dataset_abs), 'court_cache')
        name = "court_verdict_pool_{0}_{1}_{2}.pkl".format(
            dataset_key, self.pool_size, os.path.basename(self.checkpoint_path))
        if overrides:
            digest = hashlib.md5(
                repr(sorted((k, float(v)) for k, v in overrides.items())).encode('utf-8')
            ).hexdigest()[:12]
            name = "court_verdict_pool_{0}_{1}_{2}_ovr_{3}.pkl".format(
                dataset_key, self.pool_size, os.path.basename(self.checkpoint_path), digest)
        return os.path.join(cache_dir, name)

    def build_pool(self, overrides=None):
        """Пул вердиктов: выборка -> предсказание -> правовые ограничения (+переопределения) -> кэш."""
        import os
        import pickle
        cache_key = self._cache_path(overrides)
        if os.path.exists(cache_key):
            with open(cache_key, 'rb') as fh:
                self.pool = pickle.load(fh)
            return self.pool
        system = self._load_system()
        df = self.load_case_dataset()
        n = min(self.pool_size, len(df))
        sample = df.sample(n=n, random_state=42).copy()
        if sample is None or len(sample) == 0:
            raise RuntimeError(
                "Пустой пул вердиктов: выборка из датасета дел пуста (проверьте датасет дел: {0}).".format(
                    self.dataset_path))
        for col in system.feature_columns:   # заполнение отсутствующих c_* нулями (в резерве)
            if col not in sample.columns:
                sample[col] = 0
        sample = enrich_regulatory_features(sample)
        pred = system.bundle.predict(sample)
        constrained, _ = constrain_prediction_arrays(pred, sample, overrides=overrides)
        pool = []
        for i in range(len(sample)):
            row = {target: constrained[target][i] for target in constrained}
            row['c_article_code'] = int(sample.iloc[i]['c_article_code']) if 'c_article_code' in sample.columns else 128
            pool.append(remap_verdict_fields(row))
        os.makedirs(os.path.dirname(cache_key), exist_ok=True)
        with open(cache_key, 'wb') as fh:
            pickle.dump(pool, fh)
        self.pool = pool
        return pool

    def sample_verdict(self):
        """Случайная выборка вердикта из пула."""
        if self.pool is None:
            self.build_pool()
        return self.pool[np.random.randint(len(self.pool))]


def population_plan(n_active, n_sanctioned_union, n_returning):
    """Ревью 1.3 (2026-09-03): пересчёт числа активных водителей.

    n_active           — P = len(self.env): активные, сыгравшие эту эпоху
    n_sanctioned_union — C = |СВвТ ∪ СВбЛ|: НОВЫЕ санкции среди активных (объединение;
                         двойная санкция считается один раз)
    n_returning        — R: освобождённые в эту эпоху (срок истёк, нет ни в одном стеке)

    При точной консервации N = P + |J0 ∪ L0| формула N_active_new = N − |J ∪ L|_after
    сводится к N_active_new = P − C + R (доказано в spec 2026-09-03).
    """
    C, R = n_sanctioned_union, n_returning
    n_active_new = n_active - C + R
    scenario = 'grow' if R > C else ('shrink' if R < C else 'steady')
    return {'n_active_new': n_active_new, 'scenario': scenario,
            'n_survivors': n_active - C}


def breeding_plan(n_active, n_active_new, n_survivors):
    """Ревью 2.1.1/2.1.2 (вариант B): размер родительской выборки и число клонов.

    Новое поколение разводится РОВНО до n_active_new: 2*k + extra == n_active_new
    (клоны добирают чётность — дрейфа нет, решение R4 2026-09-03).
    shrink:   k_branch = floor(n_active_new / 2)   (выборка A, 2.1.1)
    grow:     k_branch = floor(n_active / 2)       (выборка B, 2.1.2)
    steady:   k_branch = floor(n_active_new / 2)
    k = min(k_branch, n_survivors); extra = n_active_new − 2*k ≥ 0 (всегда).

    Хвост консервации: если k_branch=0 (напр. P==1 в ветке grow), но цель ≥ 2 и есть
    хотя бы один выживший — разводим минимум 1 пару (k=1). Иначе ветка «прямой передачи»
    пропустила бы R вернувшихся, и сумма |env ∪ стеки| упала бы (запрещённый дрейф, R2/R4).
    """
    if n_active_new < n_active:
        k_branch = n_active_new // 2
    elif n_active_new > n_active:
        k_branch = n_active // 2
    else:
        k_branch = n_active_new // 2
    k = min(k_branch, n_survivors)
    if k == 0 and n_active_new >= 2 and n_survivors >= 1:
        k = 1
    return {'k': k, 'k_branch': k_branch, 'extra': n_active_new - 2 * k}


def select_parents(drivers, govs, dr_pay, gv_pay, surv_mask, n_select):
    """Родители — только выжившие (неосуждённые) пары; ранжирование независимое по pay.

    Осуждённые уже в стеке (ревью 1.1), поэтому «популяция» для сортировки — выжившие.
    Раньше sel_g брался по всем P (правительство осуждённого могло быть родителем);
    теперь только из пар выживших (решение R3, 2026-09-03).
    """
    surv_idx = np.where(surv_mask)[0]
    sel_d = [drivers[i] for i in sorted(surv_idx, key=lambda i: -dr_pay[i])][:n_select]
    sel_g = [govs[i] for i in sorted(surv_idx, key=lambda i: -gv_pay[i])][:n_select]
    return sel_d, sel_g


class Env:
    def __init__(self, N_drivers, fatal_min,
                 fatal_coeff, crash_speed_coeff,
                 crash_cost, spendings_tolarance_coeff,
                 chance_of_mutation, delta_maintenance, fee_tolarance_coeff, check_coeff,
                 decrease_coeff=0.8,
                 drunk_min=0.1, drunk_max=0.9,
                 theta_dr=1.0, fatal_drunk_coeff=1.0, jail_coeff_decrease=1.0, licence_coeff_decrease=1.0, conditional_decision_decrease=1.0,
                 cost_jail_per_epoch_m=361532.77509439626, cost_jail_per_epoch_s=5839841.518438178,
                 lite_g_chance=0.05, strict_g_chance=0.05,
                 court_pool=None):

        self.check_coeff = check_coeff
        self.epoch = 0
        self.fatal = []
        self.crashes = []
        self.Belted_NotBelted = []
        self.Speedings = []
        self.earns = []
        self.Strict_Lite = []
        self.averageWaiting = []
        self.averageMaint = []
        self.averageSpeed = []
        self.amountOfFatal = []
        self.averageWaitingNoFat = []
        # Метрики (раздел 3)
        self.Drunk = []          # число пьяных водителей
        self.DrunkChecked = []   # число пьяных водителей, остановленных полицией (drunk AND checked)
        self.ActivePopulation = []  # число активных водителей в эпохе (P, без тюрьмы и без лишения лицензий)
        self.Jailed = []         # число водителей в тюрьме
        self.NoLicense = []      # число водителей без лицензии

        self.lite_g_chance = lite_g_chance
        self.strict_g_chance = strict_g_chance

        self.N = N_drivers
        self.fatal_min = fatal_min
        self.crash_cost = crash_cost

        self.fatal_coeff = fatal_coeff
        self.crash_speed_coeff = crash_speed_coeff
        self.decrease_coeff = decrease_coeff  # θ_d — восприятие времени поездки при превышении
        self.spendings_coeff = spendings_tolarance_coeff

        self.chance_of_mutation = chance_of_mutation
        self.delta_maintenance = delta_maintenance
        self.fee_tolarance_coeff = fee_tolarance_coeff

        # Пьянство (1.2)
        self.drunk_min = drunk_min
        self.drunk_max = drunk_max
        self.theta_dr = theta_dr   # θ_dr — восприятие дороги пьяным водителем
        self.fatal_drunk_coeff = fatal_drunk_coeff  # f_dr — риск ДТП у пьяных
        self.jail_coeff_decrease = jail_coeff_decrease  # θ_j — лишение свободы
        self.licence_coeff_decrease = licence_coeff_decrease  # θ_l — лишение лицензии
        self.conditional_decision_decrease = conditional_decision_decrease  # θ_con — условное осуждение

        # Суд (1.2): стоимость содержания заключённого на эпоху
        self.cost_jail_per_epoch_m = cost_jail_per_epoch_m
        self.cost_jail_per_epoch_s = cost_jail_per_epoch_s

        # Стеки (1.7): СВвТ / СВбЛ
        self.jail_stack = []
        self.license_stack = []

        # Пул вердиктов суда: dict {'lite': [...], 'strict': [...]} или список (оба типа).
        if isinstance(court_pool, dict):
            self.court_pool_lite = court_pool.get('lite')
            self.court_pool_strict = court_pool.get('strict')
        else:
            self.court_pool_lite = court_pool
            self.court_pool_strict = court_pool

    def generate(self, chance_crash_s, mu_police_time_s, sigma_police_time_s, check_duration_s, check_chance_s,
                 traffic_spendings_s,
                 fee_belt_min_s, fee_belt_max_s, fee_speeding_min_s, fee_speeding_max_s, maintance_coeff_s,

                 chance_crash_l, mu_police_time_l, sigma_police_time_l, check_duration_l, check_chance_l,
                 traffic_spendings_l,
                 fee_belt_min_l, fee_belt_max_l, fee_speeding_min_l, fee_speeding_max_l, maintance_coeff_l,

                 speed_n_min, speed_n_max, speed_s_min, speed_s_max, exp_road_duration, sigma_road_duration, length,
                 chance_fatal_l, chance_fatal_s):

        self.chance_fatal_l = chance_fatal_l
        self.chance_fatal_s = chance_fatal_s

        self.exp_road_duration = exp_road_duration
        self.sigma_road_duration = sigma_road_duration
        self.length = length
        self.chance_crash_s = chance_crash_s
        self.mu_police_time_s = mu_police_time_s
        self.sigma_police_time_s = sigma_police_time_s
        self.check_duration_s = check_duration_s
        self.check_chance_s = check_chance_s
        self.traffic_spendings_s = traffic_spendings_s
        self.maintance_coeff_s = maintance_coeff_s

        self.chance_crash_l = chance_crash_l
        self.mu_police_time_l = mu_police_time_l
        self.sigma_police_time_l = sigma_police_time_l
        self.check_duration_l = check_duration_l
        self.check_chance_l = check_chance_l
        self.traffic_spendings_l = traffic_spendings_l
        self.maintance_coeff_l = maintance_coeff_l

        self.speed_n_min = speed_n_min
        self.speed_n_max = speed_n_max
        self.speed_s_min = speed_s_min
        self.speed_s_max = speed_s_max

        self.fee_belt_min_l = fee_belt_min_l
        self.fee_belt_max_l = fee_belt_max_l
        self.fee_speeding_min_l = fee_speeding_min_l
        self.fee_speeding_max_l = fee_speeding_max_l

        self.fee_belt_min_s = fee_belt_min_s
        self.fee_belt_max_s = fee_belt_max_s
        self.fee_speeding_min_s = fee_speeding_min_s
        self.fee_speeding_max_s = fee_speeding_max_s

        self.length = length
        # Матричные операции
        N = self.N
    
        isStrict = np.random.binomial(1, 0.5, N).astype(bool)
        isSpeeding = np.random.binomial(1, 0.5, N).astype(bool)
        isBelt = np.random.binomial(1, 0.5, N)
    
        fee_belt = np.where(isStrict,
                            np.random.uniform(fee_belt_min_s, fee_belt_max_s, N).astype(int),
                            np.random.uniform(fee_belt_min_l, fee_belt_max_l, N).astype(int))

        fee_speed = np.where(isStrict,
                            np.random.uniform(fee_speeding_min_s, fee_speeding_max_s, N).astype(int),
                            np.random.uniform(fee_speeding_min_l, fee_speeding_max_l, N).astype(int))

        speed = np.where(isSpeeding,
                        np.random.uniform(speed_s_min, speed_s_max, N).astype(int),
                        np.random.uniform(speed_n_min, speed_n_max, N).astype(int))

        car_crash = np.where(isStrict, chance_crash_s, chance_crash_l)
        mu_wait = np.where(isStrict, mu_police_time_s, mu_police_time_l)
        sigma_wait = np.where(isStrict, sigma_police_time_s, sigma_police_time_l)
        check_duration = np.where(isStrict, check_duration_s, check_duration_l)
        check_chance = np.where(isStrict, check_chance_s, check_chance_l)
        traffic_spendings = np.where(isStrict, traffic_spendings_s, traffic_spendings_l)
        maintance_coeff = np.where(isStrict, maintance_coeff_s, maintance_coeff_l)
        fatal_chance = np.where(isStrict, chance_fatal_s, chance_fatal_l)
    
        array_of_pairs = []

        for i in range(N):
            drunk_chance = np.random.uniform(self.drunk_min, self.drunk_max)
            driver = Driver(isSpeeding[i], speed[i], exp_road_duration,
                            sigma_road_duration, length, isBelt[i], drunk_driving_chance=drunk_chance)
            government = Goverment(isStrict[i], car_crash[i], mu_wait[i],
                                   sigma_wait[i], check_duration[i], check_chance[i],
                                   traffic_spendings[i], fee_belt[i], fee_speed[i],
                                   maintance_coeff[i], fatal_chance[i])
            array_of_pairs.append([driver, government])

        self.env = np.array(array_of_pairs)
        return array_of_pairs

    def sample_verdict_for(self, is_strict: bool = False):
        """Случайная выборка вердикта из пула соответствующего типа правительства."""
        pool = self.court_pool_strict if is_strict else self.court_pool_lite
        if pool is None or len(pool) == 0:
            raise RuntimeError("Пустой пул вердиктов: не задан court_pool (проверьте датасет дел).")
        verdict = dict(pool[np.random.randint(len(pool))])
        return verdict

    def payoff_batch_optimized(self):
        P = len(self.env)
        drivers = [pair[0] for pair in self.env]
        governments = [pair[1] for pair in self.env]

        speeds = np.array([d.speed for d in drivers])
        belts = np.array([int(d.belt) for d in drivers])
        speedings = np.array([int(d.speeding) for d in drivers])
        exps = np.array([d.exp for d in drivers])
        drunk = np.array([int(d.drunk) for d in drivers])   # 1.3: активация пьянства

        gov_fatals = np.array([g.fatal for g in governments])
        gov_crash_chances = np.array([g.crash_chance for g in governments])
        gov_mus = np.array([g.mu for g in governments])
        gov_sigmas = np.array([g.sigma for g in governments])
        gov_durations = np.array([g.duration for g in governments])
        gov_maintance_coeffs = np.array([g.maintance_coeff for g in governments])
        gov_check_chances = np.array([g.check_chance for g in governments])
        gov_fee_belts = np.array([g.fee_belt for g in governments])
        gov_fee_speedings = np.array([g.fee_speeding for g in governments])
        gov_spendings_arr = np.array([g.spendings for g in governments])
        gov_is_strict = np.array([int(g.isStrict) for g in governments])

        checks = np.random.binomial(1, gov_check_chances, P)

        # Риск ДТП/смерти растёт у пьяных одинаково: (1 + drunk * fatal_drunk_coeff)
        fatal_probs_not_speeding = gov_fatals * (1 + drunk * self.fatal_drunk_coeff) * (1 - belts * (1 - self.fatal_min))
        fatal_probs_speeding = fatal_probs_not_speeding * self.fatal_coeff
        fatals = np.where(
            speedings == 0,
            np.random.binomial(1, fatal_probs_not_speeding, P),
            np.random.binomial(1, fatal_probs_speeding, P)
        )

        crash_probs_not_speeding = gov_crash_chances * (1 + drunk * self.fatal_drunk_coeff)
        crash_probs_speeding = crash_probs_not_speeding * self.crash_speed_coeff
        crashes = np.where(
            speedings == 0,
            np.random.binomial(1, crash_probs_not_speeding, P),
            np.random.binomial(1, crash_probs_speeding, P)
        )

        waitings = np.abs(np.random.normal(gov_mus, gov_sigmas, P))

        # Время поездки: len·θ_d/(speed*θ_ma*(1+drunk*θ_dr)) при превышении скорости,
        # иначе len/(speed*θ_ma*(1+drunk*θ_dr)); θ_ma=maintance_coeff, θ_d=decrease_coeff
        # (DSS/DMS в Table 2 методологии).
        time_discount = np.where(speedings == 0, 1.0, self.decrease_coeff)
        base_time = exps - (self.length * time_discount / (speeds * gov_maintance_coeffs * (1 + drunk * self.theta_dr)))

        driver_values = base_time
        driver_values = driver_values - fatals * 1e7
        driver_values = driver_values - crashes * (waitings + gov_durations * self.check_coeff)
        driver_values = driver_values - checks * gov_durations * self.check_coeff

        # Суд (1.3-1.5): COURT = (CHECK or CRASH) and DRUNK
        court_mask = ((checks == 1) | (crashes == 1)) & (drunk == 1)
        court_jail_hours = np.zeros(P)
        court_license_hours = np.zeros(P)
        court_cond = np.zeros(P)
        court_fee_d = np.zeros(P)
        court_fee_leg = np.zeros(P)
        for i in np.where(court_mask)[0]:
            verdict = self.sample_verdict_for(is_strict=governments[i].isStrict)
            d = drivers[i]
            g = governments[i]
            d.court = True
            d.d_imprisoment_epoch = verdict['d_imprisoment_epoch']
            d.d_license_revocation_epoch = verdict['d_license_revocation_epoch']
            d.d_condition = int(verdict['d_condition'])
            d.d_fine_rub = verdict['d_fine_rub']
            d.d_reconcilation = int(verdict['d_reconcilation'])
            d.d_costs_on_state = int(verdict['d_costs_on_state'])
            d.d_costs_amount_rub = verdict['d_costs_amount_rub']
            d.d_travel_ban = int(verdict['d_travel_ban'])
            d.d_self_recusal = int(verdict['d_self_recusal'])
            # HOURS = EPOCH * 180 дней * 24 часа (1.4.1.1)
            court_jail_hours[i] = verdict['d_imprisoment_epoch'] * 180.0 * 24.0
            court_license_hours[i] = verdict['d_license_revocation_epoch'] * 180.0 * 24.0
            court_cond[i] = int(verdict['d_condition'])
            court_fee_d[i] = verdict['d_fine_rub']
            court_fee_leg[i] = verdict['d_costs_amount_rub'] if verdict['d_costs_on_state'] else 0.0
            g.feeD = verdict['d_fine_rub']
            g.feeLeg = court_fee_leg[i]

        # Фактор суда для водителя: COURT*(d_j^cou*θ_j + d_l^cou*θ_l)*(1 - COND*θ_con)
        driver_values = driver_values - (
            court_mask
            * (court_jail_hours * self.jail_coeff_decrease + court_license_hours * self.licence_coeff_decrease)
            * (1 - court_cond * self.conditional_decision_decrease)
        )

        # Метрика времени ожидания
        driver_waits_base = exps - (self.length / (speeds * gov_maintance_coeffs))
        driver_waits = driver_waits_base - fatals * 1e7
        driver_waits = driver_waits - crashes * (waitings + gov_durations)
        driver_waits = driver_waits - checks * gov_durations

        # Государство (1.5.2): (-c*Bern(crash) - tr)*θ_sp + ((1-belted)*feeB + speeding*feeS)*θ_fee*Bern(check)
        #                     + COURT*(feeD*θ_fee - feeLeg*θ_sp) - (jailed/num_drivers)*cost_jail_per_epoch*θ_sp
        gov_values = (-self.crash_cost * crashes - gov_spendings_arr) * self.spendings_coeff
        # Штраф за ремень взимается с непристёгнутых (1-belted)*feeB, как в исходной версии ver7;
        # feeS — для превышающих скорость.
        gov_values = gov_values + checks * ((1 - belts) * gov_fee_belts + speedings * gov_fee_speedings) * self.fee_tolarance_coeff
        gov_values = gov_values + (
            court_mask
            * (court_fee_d * self.fee_tolarance_coeff - court_fee_leg * self.spendings_coeff)
        )
        # Содержание заключённых (1.5.2): jailed_count из jail_stack; num_drivers =
        # не сидящие в тюрьме (в т.ч. лишённые прав) = активная P + стек лишённых прав.
        jailed_count = len(self.jail_stack)
        num_drivers = P + len(self.license_stack)
        if jailed_count > 0 and num_drivers > 0:
            c_jail_arr = np.where(gov_is_strict, self.cost_jail_per_epoch_s, self.cost_jail_per_epoch_m)
            gov_values = gov_values - (jailed_count / num_drivers) * c_jail_arr * self.spendings_coeff

        # gov_spends (метрика earns): тоже (1-belted)*feeB — согласовано с payoff
        gov_spends = (-self.crash_cost * crashes - gov_spendings_arr)
        gov_spends = gov_spends + checks * ((1 - belts) * gov_fee_belts + speedings * gov_fee_speedings)

        for i in range(P):
            drivers[i].pay = driver_values[i]
            drivers[i].fataled = bool(fatals[i])
            drivers[i].crashed = bool(crashes[i])
            drivers[i].checked = bool(checks[i])
            drivers[i].wait = driver_waits[i]
            governments[i].pay = gov_values[i]
            governments[i].spend = gov_spends[i]

    def collect_metrics_optimized(self, dr, gv):
        N = len(dr)
        
        speeds = np.array([d.speed for d in dr])
        fataled = np.array([int(d.fataled) for d in dr])
        crashed = np.array([int(d.crashed) for d in dr])
        belts = np.array([int(d.belt) for d in dr])
        speedings = np.array([int(d.speeding) for d in dr])
        waits = np.array([-d.wait for d in dr])
        
        gov_isStrict = np.array([int(g.isStrict) for g in gv])
        gov_maintance = np.array([g.maintance_coeff for g in gv])
        gov_spend = np.array([-g.spend for g in gv])
        
        # ДТП учитывает и смертельный исход: ДТП произошло, если водитель попал в ДТП
        # ИЛИ погиб за рулём (fataled). Погибшие — подмножество учтённых ДТП.
        # np.logical_or (не `|`): при пустых массивах np.array([]) имеет dtype float64,
        # и побитовый OR float не определён.
        fat = fataled.sum()
        cr = np.logical_or(crashed, fataled).sum()
        be = belts.sum()
        sp = speedings.sum()
        ea = gov_spend.sum()
        st = gov_isStrict.sum()
        mai = gov_maintance.mean() if len(gv) else 0.0
        wa = waits.sum()
        avg_speed = speeds.mean() if len(dr) else 0.0
        
        non_fatal_mask = ~fataled.astype(bool)
        wano = waits[non_fatal_mask].sum()
        bu = int(non_fatal_mask.sum())

        drunk_arr = np.array([int(d.drunk) for d in dr])
        checked_arr = np.array([int(d.checked) for d in dr])
        # Пьянство (1.3): число пьяных; пьяные, остановленные полицией (drunk AND checked)
        return {
            'fat': fat, 'cr': cr, 'be': be, 'sp': sp, 'ea': ea,
            'st': st, 'wa': wa, 'mai': mai, 'avg_speed': avg_speed,
            'wano': wano, 'bu': bu,
            'drunk_count': int(drunk_arr.sum()),
            'drunk_checked': int((drunk_arr * checked_arr).sum()),
        }

    def update_maintenance_optimized(self, gv):
        if len(gv) == 0:
            return
        gov_isStrict = np.array([g.isStrict for g in gv])
        gov_maintance = np.array([g.maintance_coeff for g in gv])
        
        strict_mask = gov_isStrict & (gov_maintance < 1)
        gov_maintance[strict_mask] += self.delta_maintenance

        lite_mask = (~gov_isStrict) & (gov_maintance > 0.3)
        gov_maintance[lite_mask] -= self.delta_maintenance
        
        for i, g in enumerate(gv):
            g.maintance_coeff = gov_maintance[i]

    def cross_drivers(self, drive, n_select=None):
        if n_select is None:
            n_select = self.N // 2
        parents = drive[0:n_select]
        # Отдельные объекты-дети: обмен не мутирует родителей и не создаёт дубликатов ссылок
        drivers = [copy.copy(d) for d in parents]
        drivers1 = [copy.copy(d) for d in parents]
        random.shuffle(drivers)
        random.shuffle(drivers1)

        for i in range(len(drivers)):
            swap_belt, swap_drunk, swap_speed = self._crossover_scenario()
            if swap_belt:
                drivers[i].belt, drivers1[i].belt = drivers1[i].belt, drivers[i].belt
            if swap_drunk:
                drivers[i].drunk_driving_chance, drivers1[i].drunk_driving_chance = (
                    drivers1[i].drunk_driving_chance, drivers[i].drunk_driving_chance)
            if swap_speed:
                # скорость и признак «превышение» меняются вместе (решение 2026-08-25)
                drivers[i].speed, drivers1[i].speed = drivers1[i].speed, drivers[i].speed
                drivers[i].speeding, drivers1[i].speeding = drivers1[i].speeding, drivers[i].speeding
        return drivers + drivers1

    def _crossover_scenario(self):
        """1.6.2.1: 7 равновероятных сценариев = непустые подмножества {belt, drunk, speed}.

        Биты: 1=belt, 2=drunk, 4=speed. randint(1,8) даёт 1..7 — каждый с p=1/7.
        """
        s = int(np.random.randint(1, 8))
        return bool(s & 1), bool(s & 2), bool(s & 4)

    def cross_gov(self, gro, n_select=None):
        if n_select is None:
            n_select = self.N // 2
        parents = gro[0: n_select]
        # Отдельные объекты-дети (как в cross_drivers): обмен не мутирует родителей
        gov = [copy.copy(g) for g in parents]
        gov1 = [copy.copy(g) for g in parents]
        random.shuffle(gov)
        random.shuffle(gov1)
        count_pr = 0

        for i in range(len(gov)):
            count_pr += gov[i].isStrict
        # print(count_pr, len(gov) - count_pr)
        for i in range(len(gov)):
            speed = bool(np.random.binomial(1, 0.5))
            belt = bool(np.random.binomial(1, 0.5))
            maint = bool(np.random.binomial(1, 0.5))
            if (speed or belt or maint) == False:
                value = np.random.randint(1, 3, 1)[0]
                if value == 1:
                    speed = True
                elif value == 2:
                    belt = True
                else:
                    maint = True

            if speed:
                if gov[i].isStrict == gov1[i].isStrict:
                    gov[i].fee_speeding, gov1[i].fee_speeding = gov1[i].fee_speeding, gov[i].fee_speeding
                else:
                    g, g1 = gov[i].fee_speeding, gov1[i].fee_speeding
                    if gov[i].isStrict:
                        gov1[i].fee_speeding = ((g - self.fee_speeding_min_s) / (
                                    self.fee_speeding_max_s - self.fee_speeding_min_s)) * (
                                                           self.fee_speeding_max_l - self.fee_speeding_min_l) + self.fee_speeding_min_l
                        gov[i].fee_speeding = ((g1 - self.fee_speeding_min_l) / (
                                    self.fee_speeding_max_l - self.fee_speeding_min_l)) * (
                                                          self.fee_speeding_max_s - self.fee_speeding_min_s) + self.fee_speeding_min_s
                    else:
                        gov1[i].fee_speeding = ((g - self.fee_speeding_min_l) / (
                                    self.fee_speeding_max_l - self.fee_speeding_min_l)) * (
                                                           self.fee_speeding_max_s - self.fee_speeding_min_s) + self.fee_speeding_min_s
                        gov[i].fee_speeding = ((g1 - self.fee_speeding_min_s) / (
                                    self.fee_speeding_max_s - self.fee_speeding_min_s)) * (
                                                          self.fee_speeding_max_l - self.fee_speeding_min_l) + self.fee_speeding_min_l

            if belt:
                if gov[i].isStrict == gov1[i].isStrict:
                    gov[i].fee_belt, gov1[i].fee_belt = gov1[i].fee_belt, gov[i].fee_belt
                else:
                    g, g1 = gov[i].fee_belt, gov1[i].fee_belt
                    if gov[i].isStrict:
                        gov1[i].fee_belt = ((g - self.fee_belt_min_s) / (self.fee_belt_max_s - self.fee_belt_min_s)) * (
                                    self.fee_belt_max_l - self.fee_belt_min_l) + self.fee_belt_min_l
                        gov[i].fee_belt = ((g1 - self.fee_belt_min_l) / (self.fee_belt_max_l - self.fee_belt_min_l)) * (
                                    self.fee_belt_max_s - self.fee_belt_min_s) + self.fee_belt_min_s
                    else:
                        gov1[i].fee_belt = ((g - self.fee_belt_min_l) / (self.fee_belt_max_l - self.fee_belt_min_l)) * (
                                    self.fee_belt_max_s - self.fee_belt_min_s) + self.fee_belt_min_s
                        gov[i].fee_belt = ((g1 - self.fee_belt_min_s) / (self.fee_belt_max_s - self.fee_belt_min_s)) * (
                                    self.fee_belt_max_l - self.fee_belt_min_l) + self.fee_belt_min_l

            if maint:
                gov[i].maintance_coeff, gov1[i].maintance_coeff = gov1[i].maintance_coeff, gov[i].maintance_coeff

        gove = gov + gov1
        return gove

    def mutation_driver(self, d):
        # 1.6.3.1: мутация шанса пьянства -> [drunk_min; drunk_max]
        if np.random.binomial(1, self.chance_of_mutation):
            d.drunk_driving_chance = np.random.uniform(self.drunk_min, self.drunk_max)
        speeding = bool(np.random.binomial(1, self.chance_of_mutation))
        belt = bool(np.random.binomial(1, self.chance_of_mutation))
        norm = bool(np.random.binomial(1, self.chance_of_mutation))
        fast = bool(np.random.binomial(1, self.chance_of_mutation))

        if speeding:
            if d.speeding:
                d.speed = np.random.uniform(self.speed_s_min, self.speed_s_max)
            else:
                d.speed = np.random.uniform(self.speed_n_min, self.speed_n_max)
        if belt:
            d.belt = not speeding
        if fast:
            fast_speed = np.random.uniform(self.speed_s_min, self.speed_s_max)
            isBelt = np.random.binomial(1, 0.5)
            d = Driver(True, fast_speed, self.exp_road_duration, self.sigma_road_duration, self.length, d.belt,
                       drunk_driving_chance=d.drunk_driving_chance)
        elif norm:
            norm_speed = np.random.uniform(self.speed_n_min, self.speed_n_max)
            isBelt = np.random.binomial(1, 0.5)
            d = Driver(False, norm_speed, self.exp_road_duration, self.sigma_road_duration, self.length, d.belt,
                       drunk_driving_chance=d.drunk_driving_chance)
        return d

    def mutation_gov(self, g):
        speed = bool(np.random.binomial(1, self.chance_of_mutation))
        belt = bool(np.random.binomial(1, self.chance_of_mutation))
        maint = bool(np.random.binomial(1, self.chance_of_mutation))
        srict = bool(np.random.binomial(1, self.strict_g_chance))
        lite = bool(np.random.binomial(1, self.lite_g_chance))

        if speed:
            if g.isStrict:
                g.fee_speeding = np.random.uniform(self.fee_speeding_min_s, self.fee_speeding_max_s)
            else:
                g.fee_speeding = np.random.uniform(self.fee_speeding_min_l, self.fee_speeding_max_l)

        if belt:
            if g.isStrict:
                g.fee_belt = np.random.uniform(self.fee_belt_min_s, self.fee_belt_max_s)
            else:
                g.fee_belt = np.random.uniform(self.fee_belt_min_l, self.fee_belt_max_l)

        if maint:
            g.maintance_coeff = np.random.uniform(0.4, 1)

        if srict:
            cost_belt_fee = np.random.uniform(self.fee_belt_min_s, self.fee_belt_max_s)
            cost_speeding_fee = np.random.uniform(self.fee_speeding_min_s, self.fee_speeding_max_s)

            g = Goverment(True, self.chance_crash_s, self.mu_police_time_s, self.sigma_police_time_s,
                          self.check_duration_s, self.check_chance_s, self.traffic_spendings_s,
                          cost_belt_fee, cost_speeding_fee, self.maintance_coeff_s, self.chance_fatal_s)
        elif lite:
            cost_belt_fee = np.random.uniform(self.fee_belt_min_l, self.fee_belt_max_l)
            cost_speeding_fee = np.random.uniform(self.fee_speeding_min_l, self.fee_speeding_max_l)

            g = Goverment(False, self.chance_crash_l, self.mu_police_time_l, self.sigma_police_time_l,
                          self.check_duration_l, self.check_chance_l, self.traffic_spendings_l,
                          cost_belt_fee, cost_speeding_fee, self.maintance_coeff_l, self.chance_fatal_l)

        return g

    def update_stacks(self, convicted_drivers, no_license_drivers):
        """Стеки (1.7): СВвТ/СВбЛ. Уменьшение таймеров, добавление, возврат освобождённых.

        Водитель с двойной санкцией возвращается, только когда его нет ни в одном стеке.
        """
        released_jail = [d for (d, e) in self.jail_stack if e - 1 <= 0]
        self.jail_stack = [(d, e - 1) for (d, e) in self.jail_stack if e - 1 > 0]
        released_license = [d for (d, e) in self.license_stack if e - 1 <= 0]
        self.license_stack = [(d, e - 1) for (d, e) in self.license_stack if e - 1 > 0]
        for d in convicted_drivers:
            self.jail_stack.append((d, d.d_imprisoment_epoch))
        for d in no_license_drivers:
            self.license_stack.append((d, d.d_license_revocation_epoch))
        still_frozen = {d for d, _ in self.jail_stack} | {d for d, _ in self.license_stack}
        returning = [d for d in set(released_jail) | set(released_license) if d not in still_frozen]
        return returning

    def generate_new_government(self):
        """Случайное правительство (для водителя, вернувшегося из стека)."""
        isStrict = bool(np.random.binomial(1, 0.5))
        if isStrict:
            fee_belt = np.random.uniform(self.fee_belt_min_s, self.fee_belt_max_s)
            fee_speed = np.random.uniform(self.fee_speeding_min_s, self.fee_speeding_max_s)
            crash = self.chance_crash_s
            mu_wait = self.mu_police_time_s
            sigma_wait = self.sigma_police_time_s
            check_dur = self.check_duration_s
            check_chance = self.check_chance_s
            spendings = self.traffic_spendings_s
            maint = self.maintance_coeff_s
            fatal = self.chance_fatal_s
        else:
            fee_belt = np.random.uniform(self.fee_belt_min_l, self.fee_belt_max_l)
            fee_speed = np.random.uniform(self.fee_speeding_min_l, self.fee_speeding_max_l)
            crash = self.chance_crash_l
            mu_wait = self.mu_police_time_l
            sigma_wait = self.sigma_police_time_l
            check_dur = self.check_duration_l
            check_chance = self.check_chance_l
            spendings = self.traffic_spendings_l
            maint = self.maintance_coeff_l
            fatal = self.chance_fatal_l
        return Goverment(isStrict, crash, mu_wait, sigma_wait, check_dur, check_chance,
                         spendings, fee_belt, fee_speed, maint, fatal)

    def run_epoch_optimized(self):
        # 1. Активация пьянства (1.3); COURT сбрасывается в FALSE в начале эпохи
        for pair in self.env:
            d = pair[0]
            d.drunk = bool(np.random.binomial(1, d.drunk_driving_chance))
            d.court = False
            d.d_imprisoment_epoch = 0
            d.d_license_revocation_epoch = 0
            d.d_condition = 0
            d.d_fine_rub = 0.0
            d.d_reconcilation = 0
            d.d_costs_on_state = 0
            d.d_costs_amount_rub = 0.0
            d.d_travel_ban = 0
            d.d_self_recusal = 0

        # 2. Payoff (1.3-1.5)
        self.payoff_batch_optimized()

        P = len(self.env)
        dr = [self.env[i][0] for i in range(P)]
        gv = [self.env[i][1] for i in range(P)]

        # 3. Метрики (нормировка на активную P; при P==0 доли 0)
        metrics = self.collect_metrics_optimized(dr, gv)
        self.fatal.append(metrics['fat'])
        self.crashes.append(metrics['cr'])
        self.Belted_NotBelted.append(metrics['be'] / P if P else 0.0)
        self.Speedings.append(metrics['sp'] / P if P else 0.0)
        self.Strict_Lite.append(metrics['st'] / P if P else 0.0)
        self.earns.append(metrics['ea'] / P if P else 0.0)
        self.averageWaitingNoFat.append((metrics['wano'] / metrics['bu']) * 60 if metrics['bu'] > 0 else 0)
        self.averageWaiting.append(metrics['wa'] / P if P else 0.0)
        self.averageMaint.append(metrics['mai'])
        self.averageSpeed.append(metrics['avg_speed'])
        self.Drunk.append(metrics['drunk_count'])
        self.DrunkChecked.append(metrics['drunk_checked'])
        self.ActivePopulation.append(P)

        # 4. Обслуживание (1.8)
        self.update_maintenance_optimized(gv)

        # 5. Санкции эпохи (1.3): осуждённые (СВвТ) и лишённые прав (СВбЛ) среди активных.
        convicted_drivers = [dr[i] for i in range(P) if dr[i].d_imprisoment_epoch > 0]
        no_license_drivers = [dr[i] for i in range(P) if dr[i].d_license_revocation_epoch > 0]
        convicted = np.array([d.d_imprisoment_epoch > 0 for d in dr], dtype=bool)
        no_license = np.array([d.d_license_revocation_epoch > 0 for d in dr], dtype=bool)
        C_new = int(np.logical_or(convicted, no_license).sum())   # объединение (1.3), не сумма

        # 6. Стеки (1.7): уменьшение сроков, добавление новых приговоров (ревью 1.1/1.2).
        released = self.update_stacks(convicted_drivers, no_license_drivers)
        self.Jailed.append(len(self.jail_stack))
        self.NoLicense.append(len(self.license_stack))

        # 7. Пересчёт активных (ревью 1.3), вариант B (решение 2026-09-03): новое поколение
        #    разводится РОВНО до N_active_new = P − C + R. Освобождённые не возвращаются —
        #    их тождество не продолжается; слоты заполняют дети/клоны выживших.
        plan = population_plan(P, C_new, len(released))
        n_active_new, S = plan['n_active_new'], plan['n_survivors']

        # 8. Отбор (2.1.1/2.1.2): родители — только выжившие пары (R3).
        dr_pay = np.array([d.pay for d in dr])
        gv_pay = np.array([g.pay for g in gv])
        surv_mask = ~convicted & ~no_license
        b = breeding_plan(P, n_active_new, S)
        if n_active_new == 0 or S == 0:
            # вырожденно (R5): разводить некому — вернувшиеся остаются взрослыми (их ровно
            # N_active_new = R), иначе популяция исчезла бы навсегда.
            env1 = [[d, self.generate_new_government()] for d in released]
        elif b['k'] == 0:
            # хвост: цель == 1 и есть ровно один выживший (R == 0) — прямая передача.
            env1 = [[dr[i], gv[i]] for i in range(P) if surv_mask[i]]
        else:
            sel_d, sel_g = select_parents(dr, gv, dr_pay, gv_pay, surv_mask, b['k'])
            child_d = self.cross_drivers(sel_d, b['k'])   # ровно 2k детей
            child_g = self.cross_gov(sel_g, b['k'])
            if b['extra'] > 0:
                # 2.1.2: delta = N_active_new − 2k клонов равномерно из выборки (чётность, R4)
                pick = np.random.randint(0, b['k'], size=b['extra'])
                for j in pick:
                    child_d.append(copy.copy(sel_d[j]))
                    child_g.append(copy.copy(sel_g[j]))
            # каждый ребёнок — отдельный объект: copy.copy защищает консервацию |env ∪ стеки|
            dr1 = [copy.copy(self.mutation_driver(d)) for d in child_d]
            gv1 = [self.mutation_gov(g) for g in child_g]
            env1 = [[dr1[i], gv1[i]] for i in range(len(gv1))]

        self.epoch = self.epoch + 1
        self.env = env1

    run_epoch = run_epoch_optimized


# Один эксперимент в отдельном процессе.
def run_single_experiment(params):
    (k, N_runs, N_epochs, N_drivers, fatal_belt_min,
     fatal_coeff, crash_speed_coeff, crash_cost, spendings_tolarance_coeff,
     chance_of_mutation, delta_maintenance, fee_tolarance_coeff, check_coeff, decrease_coeff,
     chance_crash_s, mu_police_time_s, sigma_police_time_s, check_duration_s, check_chance_s,
     traffic_spendings_s, fee_belt_min_s, fee_belt_max_s, fee_speeding_min_s,
     fee_speeding_max_s, maintance_coeff_s,
     chance_crash_l, mu_police_time_l, sigma_police_time_l, check_duration_l, check_chance_l,
     traffic_spendings_l, fee_belt_min_l, fee_belt_max_l, fee_speeding_min_l,
     fee_speeding_max_l, maintance_coeff_l,
     speed_n_min, speed_n_max, speed_s_min, speed_s_max, exp_road_duration,
     sigma_road_duration, length, chance_fatal_l, chance_fatal_s,
     drunk_min, drunk_max, theta_dr, fatal_drunk_coeff, jail_coeff_decrease, licence_coeff_decrease, conditional_decision_decrease,
     cost_jail_per_epoch_m, cost_jail_per_epoch_s, lite_g_chance, strict_g_chance,
     court_pool) = params

    env = Env(N_drivers, fatal_belt_min, fatal_coeff, crash_speed_coeff,
              crash_cost, spendings_tolarance_coeff, chance_of_mutation,
              delta_maintenance, fee_tolarance_coeff, check_coeff, decrease_coeff,
              drunk_min, drunk_max, theta_dr, fatal_drunk_coeff, jail_coeff_decrease, licence_coeff_decrease, conditional_decision_decrease,
              cost_jail_per_epoch_m, cost_jail_per_epoch_s, lite_g_chance, strict_g_chance,
              court_pool)

    env.generate(chance_crash_s, mu_police_time_s, sigma_police_time_s, check_duration_s, check_chance_s,
                 traffic_spendings_s,
                 fee_belt_min_s, fee_belt_max_s, fee_speeding_min_s, fee_speeding_max_s, maintance_coeff_s,
                 chance_crash_l, mu_police_time_l, sigma_police_time_l, check_duration_l, check_chance_l,
                 traffic_spendings_l,
                 fee_belt_min_l, fee_belt_max_l, fee_speeding_min_l, fee_speeding_max_l, maintance_coeff_l,
                 speed_n_min, speed_n_max, speed_s_min, speed_s_max, exp_road_duration,
                 sigma_road_duration, length, chance_fatal_l, chance_fatal_s)

    for i in range(N_epochs):
        env.run_epoch()
    return {
        'run_id': k,
        'fatal': env.fatal,
        'crashes': env.crashes,
        'Belted_NotBelted': env.Belted_NotBelted,
        'Speedings': env.Speedings,
        'earns': env.earns,
        'Strict_Lite': env.Strict_Lite,
        'averageWaiting': env.averageWaiting,
        'averageMaint': env.averageMaint,
        'averageSpeed': env.averageSpeed,
        'amountOfFatal': env.amountOfFatal,
        'averageWaitingNoFat': env.averageWaitingNoFat,
        'Drunk': env.Drunk,
        'DrunkChecked': env.DrunkChecked,
        'ActivePopulation': env.ActivePopulation,
        'Jailed': env.Jailed,
        'NoLicense': env.NoLicense,
    }


def add_arr_parallel(results_list):
    # Суммирование результатов экспериментов
    if not results_list:
        return None
    
    accumulated = {key: [0] * len(results_list[0][key]) for key in results_list[0].keys() if key != 'run_id'}
    
    for result in results_list:
        for key in accumulated.keys():
            for i in range(len(accumulated[key])):
                accumulated[key][i] += result[key][i]
    
    return accumulated


def add_mean_parallel(accumulated, n):
    mean_result = {}
    for key in accumulated.keys():
        mean_result[key] = [x / n for x in accumulated[key]]

    return mean_result


# ---------------------------------------------------------------------------
# Метаданные для экспорта: листы «meta» и «exp_epoch»
# Параметры и их смысл описаны в «Параметры запуска.xlsx», методологии MVP
# и статье (Tables 1-4). Лист «meta» = группа «Мета-параметры» (Table 2),
# лист «exp_epoch» = группа «Параметры водителей» (в статье "exp" — ожидаемая
# длительность поездки exp_road_duration, используется в каждой эпохе).
# ---------------------------------------------------------------------------


def build_param_frame(params, group_names, run_info=None, comp_info=None):
    """Таблица параметров для листа экспорта.

    group_names — группы из _META_ROWS, включаемые в таблицу (например,
    ['Мета-параметры'] или ['Параметры водителей']).
    run_info/comp_info — списки кортежей (ключ, значение) с информацией о запуске;
    добавляются в конце.
    """
    rows = []
    for group, key, unit, _desc in _META_ROWS:
        if group not in group_names or key not in params:
            continue
        rows.append({'Группа': group, 'Параметр': key,
                     'Значение': params[key], 'Единица': unit})
    for info_rows, group in ((run_info, 'Информация о запуске'),
                             (comp_info, 'Информация о запуске (2-й прогон)')):
        if not info_rows:
            continue
        for key, value in info_rows:
            rows.append({'Группа': group, 'Параметр': key,
                         'Значение': value, 'Единица': ''})
    return pd.DataFrame(rows, columns=['Группа', 'Параметр', 'Значение', 'Единица'])


def _build_meta_rows():
    """Строки (группа, ключ, единица, описание) для листа «meta»."""
    rows = []

    def add(group, key, unit, desc):
        rows.append((group, key, unit, desc))

    # Мета-параметры (Table 2 статьи)
    for key, unit, desc in [
        ('N_runs', 'шт', 'Число экспериментов'),
        ('N_epochs', 'шт', 'Число эпох в одном эксперименте'),
        ('N_drivers', 'чел', 'Число водителей в эксперименте'),
        ('crash_cost', 'руб', 'Средняя стоимость ДТП'),
        ('fatal_belt_min', '', 'Коэффициент снижения фатального исхода при ремне'),
        ('delta_maintenance', '', 'Скорость деградации инфраструктуры'),
        ('random_seed', '', 'Сид случайной генерации (пусто = случайный)'),
    ]:
        add('Мета-параметры', key, unit, desc)

    # Параметры водителей (Table 4 статьи + лист Drivers)
    for key, unit, desc in [
        ('speed_n_min', 'км/ч', 'Левая граница скорости для соблюдающих'),
        ('speed_n_max', 'км/ч', 'Правая граница скорости для соблюдающих'),
        ('speed_s_min', 'км/ч', 'Левая граница скорости для превышающих'),
        ('speed_s_max', 'км/ч', 'Правая граница скорости для превышающих'),
        ('exp_road_duration', 'час', 'Субъективное ожидаемое время поездки'),
        ('sigma_road_duration', 'час', 'СКО субъективного времени поездки'),
        ('length', 'км', 'Фактическая длина пути'),
        ('drunk_min', 'шанс', 'Левая граница шанса вождения в нетрезвом виде'),
        ('drunk_max', 'шанс', 'Правая граница шанса вождения в нетрезвом виде'),
    ]:
        add('Параметры водителей', key, unit, desc)

    # Национальные коэффициенты (Table 3 статьи + методология MVP)
    for key, unit, desc in [
        ('fatal_coeff', '', 'Повышающий коэффициент фатального исхода при превышении'),
        ('crash_speed_coeff', '', 'Повышающий коэффициент ДТП при превышении'),
        ('decrease_coeff', '', 'Понижающий коэффициент восприятия времени поездки при превышении (θ_d)'),
        ('spendings_tolarance_coeff', '', 'Коэффициент субъективного восприятия расходов'),
        ('fee_tolarance_coeff', '', 'Коэффициент уплаты штрафов водителями'),
        ('chance_of_mutation', '', 'Вероятность мутации агента'),
        ('check_coeff', '', 'Коэффициент субъективного восприятия времени проверки'),
        ('theta_dr', '', 'Понижающий коэффициент восприятия дороги при пьяном вождении (θ_dr)'),
        ('fatal_drunk_coeff', '', 'Повышающий коэффициент шанса ДТП при пьяном вождении (f_dr)'),
        ('jail_coeff_decrease', '', 'Понижающий коэффициент времени лишения свободы (θ_j)'),
        ('licence_coeff_decrease', '', 'Понижающий коэффициент времени лишения прав (θ_l)'),
        ('conditional_decision_decrease', '', 'Понижающий коэффициент условного осуждения (θ_con)'),
        ('lite_g_chance', 'шанс', 'Шанс мягкого государства при мутации'),
        ('strict_g_chance', 'шанс', 'Шанс строгого государства при мутации'),
    ]:
        add('Национальные коэффициенты', key, unit, desc)

    # Параметры правительств (лист Government)
    gov_desc = [
        ('chance_crash', 'Шанс ДТП', 'шанс'),
        ('chance_fatal', 'Шанс фатального ДТП', 'шанс'),
        ('mu_police_time', 'Среднее время прибытия полиции', 'час'),
        ('sigma_police_time', 'СКО времени прибытия полиции', 'час'),
        ('check_duration', 'Длительность проверки', 'час'),
        ('check_chance', 'Шанс проверки', 'шанс'),
        ('traffic_spendings', 'Расходы на реализацию политики', 'руб'),
        ('fee_belt_min', 'Левая граница штрафа за ремень', 'руб'),
        ('fee_belt_max', 'Правая граница штрафа за ремень', 'руб'),
        ('fee_speeding_min', 'Левая граница штрафа за превышение', 'руб'),
        ('fee_speeding_max', 'Правая граница штрафа за превышение', 'руб'),
        ('maintance_coeff', 'Коэффициент инфраструктуры', ''),
    ]
    for base, desc, unit in gov_desc:
        add('Строгое правительство', base + '_s', unit, desc + ' (строгое)')
        add('Мягкое правительство', base + '_l', unit, desc + ' (мягкое)')
    add('Строгое правительство', 'cost_jail_per_epoch_s', 'руб',
        'Стоимость содержания заключённого на эпоху (строгое)')
    add('Мягкое правительство', 'cost_jail_per_epoch_m', 'руб',
        'Стоимость содержания заключённого на эпоху (мягкое)')

    # Параметры суда
    for key, unit, desc in [
        ('court_checkpoint', '', 'Путь к чекпоинту ML-модели суда'),
        ('court_dataset', '', 'Датасет пула вердиктов суда'),
        ('court_pool_size', 'шт', 'Размер пула вердиктов суда'),
    ]:
        add('Параметры суда', key, unit, desc)

    # Регуляция ст. 12.8 / 264.1 (методология MVP + лист Government)
    legal_desc = [
        ('min_licence_12.8_1', 'Мин. срок лишения прав по ст. 12.8', 'год'),
        ('max_licence_12.8_1', 'Макс. срок лишения прав по ст. 12.8', 'год'),
        ('max_fee_12.8_1', 'Макс. штраф по ст. 12.8', 'руб'),
        ('min_forced_job_264.1', 'Мин. срок принудительных работ по ст. 264.1', 'год'),
        ('max_forced_job_264.1', 'Макс. срок принудительных работ по ст. 264.1', 'год'),
        ('min_jail_264.1', 'Мин. срок лишения свободы по ст. 264.1', 'год'),
        ('max_jail_264.1', 'Макс. срок лишения свободы по ст. 264.1', 'год'),
        ('min_licence_264.1', 'Мин. срок лишения прав по ст. 264.1', 'год'),
        ('max_licence_264.1', 'Макс. срок лишения прав по ст. 264.1', 'год'),
        ('min_fee_264.1', 'Мин. штраф по ст. 264.1', 'руб'),
        ('max_fee_264.1', 'Макс. штраф по ст. 264.1', 'руб'),
    ]
    for base, desc, unit in legal_desc:
        add('Регуляция ст. 12.8 / 264.1', base + '_l', unit, desc + ' (мягкое)')
        add('Регуляция ст. 12.8 / 264.1', base + '_s', unit, desc + ' (строгое)')

    return rows


_META_ROWS = _build_meta_rows()

# Параметрический сценарий: какие параметры можно сканировать.
# Исключаем строки (seed/чекпоинт/датасет), контроль запуска (N_runs/N_epochs)
# и размер пула суда (держит поведение суда фиксированным).
SWEEP_EXCLUDED = {'N_runs', 'N_epochs', 'random_seed',
                  'court_checkpoint', 'court_dataset', 'court_pool_size'}
SWEEPABLE_PARAMS = [key for (_g, key, _u, _d) in _META_ROWS if key not in SWEEP_EXCLUDED]
# Подписи параметров — ТОЧНО как в форме настроек (единый источник для формы
# настроек и подсказок параметрического сценария, чтобы они не расходились).
PARAM_LABELS = {
    # Мета-параметры
    'N_runs': 'Количество экспериментов',
    'N_epochs': 'Количество эпох эволюции',
    'N_drivers': 'Количество водителей',
    'crash_cost': 'Стоимость аварии',
    'fatal_belt_min': 'Минимальная защита ремня',
    'delta_maintenance': 'Изменение коэф. обслуживания',
    'random_seed': 'Случайное зерно (пусто=случайно)',

    # Параметры водителей
    'speed_n_min': 'Мин. нормальная скорость',
    'speed_n_max': 'Макс. нормальная скорость',
    'speed_s_min': 'Мин. скорость при превышении',
    'speed_s_max': 'Макс. скорость при превышении',
    'exp_road_duration': 'Ожидаемая длительность дороги',
    'sigma_road_duration': 'Стд. откл. длительности',
    'length': 'Длина дороги',
    'drunk_min': 'Мин. шанс пьянства',
    'drunk_max': 'Макс. шанс пьянства',

    # Национальные коэффициенты
    'fatal_coeff': 'Коэф. штрафа за смерть',
    'crash_speed_coeff': 'Коэф. штрафа за аварию',
    'decrease_coeff': 'Коэффициент уменьшения (θ_d)',
    'spendings_tolarance_coeff': 'Коэф. толерантности расходов',
    'fee_tolarance_coeff': 'Коэф. толерантности штрафов',
    'chance_of_mutation': 'коэффициент мутации',
    'check_coeff': 'Коэффициент проверки',
    'lite_g_chance': 'Шанс мутации в мягкое',
    'strict_g_chance': 'Шанс мутации в строгое',
    'theta_dr': 'Коэф. восприятия дороги (θ_dr)',
    'fatal_drunk_coeff': 'Коэф. шанса ДТП (f_dr)',
    'jail_coeff_decrease': 'Коэф. времени лишения свободы (θ_j)',
    'licence_coeff_decrease': 'Коэф. времени лишения лицензии (θ_l)',
    'conditional_decision_decrease': 'Коэф. условного осуждения (θ_con)',

    # Параметры строгого правительства
    'chance_crash_s': 'Вероятность аварии',
    'chance_fatal_s': 'Вероятность смерти',
    'mu_police_time_s': 'Среднее время ожидания',
    'sigma_police_time_s': 'Стд. откл. времени ожидания',
    'check_duration_s': 'Длительность проверки',
    'check_chance_s': 'Вероятность проверки',
    'traffic_spendings_s': 'Транспортные расходы',
    'fee_belt_min_s': 'Мин. штраф за ремень',
    'fee_belt_max_s': 'Макс. штраф за ремень',
    'fee_speeding_min_s': 'Мин. штраф за скорость',
    'fee_speeding_max_s': 'Макс. штраф за скорость',
    'maintance_coeff_s': 'Коэффициент обслуживания',
    'cost_jail_per_epoch_s': 'Содержание заключённого (cost_jail_per_epoch)',

    # Параметры мягкого правительства
    'chance_crash_l': 'Вероятность аварии',
    'chance_fatal_l': 'Вероятность смерти',
    'mu_police_time_l': 'Среднее время ожидания',
    'sigma_police_time_l': 'Стд. откл. времени ожидания',
    'check_duration_l': 'Длительность проверки',
    'check_chance_l': 'Вероятность проверки',
    'traffic_spendings_l': 'Транспортные расходы',
    'fee_belt_min_l': 'Мин. штраф за ремень',
    'fee_belt_max_l': 'Макс. штраф за ремень',
    'fee_speeding_min_l': 'Мин. штраф за скорость',
    'fee_speeding_max_l': 'Макс. штраф за скорость',
    'maintance_coeff_l': 'Коэффициент обслуживания',
    'cost_jail_per_epoch_m': 'Содержание заключённого (cost_jail_per_epoch)',

    # Параметры суда
    'court_pool_size': 'Размер пула вердиктов (реализация)',
    'court_checkpoint': 'Checkpoint модели (реализация)',
    'court_dataset': 'Датасет дел (CSV)',

    # Регуляция ст. 12.8 / 264.1 (мягкое и строгое — одинаковая подпись)
    'min_licence_12.8_1_l': 'Лишение прав, мин. (12.8)',
    'min_licence_12.8_1_s': 'Лишение прав, мин. (12.8)',
    'max_licence_12.8_1_l': 'Лишение прав, макс. (12.8)',
    'max_licence_12.8_1_s': 'Лишение прав, макс. (12.8)',
    'max_fee_12.8_1_l': 'Штраф, макс. (12.8)',
    'max_fee_12.8_1_s': 'Штраф, макс. (12.8)',
    'min_forced_job_264.1_l': 'Принуд. работы, мин. (264.1)',
    'min_forced_job_264.1_s': 'Принуд. работы, мин. (264.1)',
    'max_forced_job_264.1_l': 'Принуд. работы, макс. (264.1)',
    'max_forced_job_264.1_s': 'Принуд. работы, макс. (264.1)',
    'min_jail_264.1_l': 'Лишение свободы, мин. (264.1)',
    'min_jail_264.1_s': 'Лишение свободы, мин. (264.1)',
    'max_jail_264.1_l': 'Лишение свободы, макс. (264.1)',
    'max_jail_264.1_s': 'Лишение свободы, макс. (264.1)',
    'min_licence_264.1_l': 'Лишение прав, мин. (264.1)',
    'min_licence_264.1_s': 'Лишение прав, мин. (264.1)',
    'max_licence_264.1_l': 'Лишение прав, макс. (264.1)',
    'max_licence_264.1_s': 'Лишение прав, макс. (264.1)',
    'min_fee_264.1_l': 'Штраф, мин. (264.1)',
    'min_fee_264.1_s': 'Штраф, мин. (264.1)',
    'max_fee_264.1_l': 'Штраф, макс. (264.1)',
    'max_fee_264.1_s': 'Штраф, макс. (264.1)',
}


def generate_sweep_values(start, end, step):
    """Закрытый интервал [start, end] с шагом step (включая конечную точку).

    Последнее значение принудительно равно end (без дрейфа плавающей точки).
    Бросает ValueError с русским сообщением при неверных границах.
    """
    start = float(start)
    end = float(end)
    step = float(step)
    if not (np.isfinite(start) and np.isfinite(end) and np.isfinite(step)):
        raise ValueError("Начало, конец и шаг должны быть конечными числами")
    if step <= 0:
        raise ValueError("Шаг должен быть больше нуля")
    if end < start:
        raise ValueError("Конец диапазона должен быть не меньше начала")
    count = int(round((end - start) / step)) + 1
    if count < 1:
        return []
    values = [start + i * step for i in range(count)]
    values[-1] = end
    return values


def format_param_value(value):
    """Кратчайшее строковое представление значения для имени файла.

    Использует str(float(...)) — обратимое представление без потери точности,
    поэтому разные значения всегда дают разные имена файлов. Хвостовой '.0'
    убирается (60.0 -> '60').
    """
    s = str(float(value))
    if 'e' not in s and '.' in s:
        s = s.rstrip('0').rstrip('.')
    return s


def sweep_filename(param1, value1, param2, value2):
    """Имя файла эксперимента: experiment_<p1>=<v1>_<p2>=<v2>.xlsx."""
    return "experiment_%s=%s_%s=%s.xlsx" % (param1, format_param_value(value1),
                                            param2, format_param_value(value2))


# Колонки сводки экспорта: (имя в файле, атрибут Env с по-эпоховыми списками).
_METRIC_COLUMNS = [
    ('fatal', 'fatal'), ('crashes', 'crashes'), ('belted', 'Belted_NotBelted'),
    ('speeding', 'Speedings'), ('earnings', 'earns'), ('strict', 'Strict_Lite'),
    ('avgWait', 'averageWaiting'), ('avgWaitNoFatal', 'averageWaitingNoFat'),
    ('avgMaint', 'averageMaint'), ('avgSpeed', 'averageSpeed'),
    ('drunk', 'Drunk'), ('drunk_checked', 'DrunkChecked'),
    ('active_population', 'ActivePopulation'), ('jailed', 'Jailed'),
    ('no_license', 'NoLicense'),
]


def _tail_mean(series, start=10):
    """Среднее по эпохам 11..N (индексы >= start). None, если эпох мало."""
    arr = np.asarray(series, dtype=float)
    if arr.size > start:
        return float(arr[start:].mean())
    return None


def build_export_frames(params, env_results, comparison_params=None, comparison_env=None):
    """Два кадра экспорта: «meta» и «exp_epoch».

    meta: первая строка — имена параметров в системе, вторая (и далее в режиме
    сравнения) — значения параметров набора. exp_epoch: ПОЛНЫЙ массив динамики
    по эпохам (каждая строка — одна эпоха, каждая колонка — метрика).
    """
    if comparison_params is not None and comparison_env is not None:
        meta_df = pd.DataFrame(
            [list(params.values()), list(comparison_params.values())],
            columns=list(params.keys()))
    else:
        meta_df = pd.DataFrame([list(params.values())], columns=list(params.keys()))

    if comparison_env is not None:
        di1 = {col + '_1': list(getattr(env_results, attr)) for col, attr in _METRIC_COLUMNS}
        di2 = {col + '_2': list(getattr(comparison_env, attr)) for col, attr in _METRIC_COLUMNS}
        di = {**di1, **di2}
    else:
        di = {col: list(getattr(env_results, attr)) for col, attr in _METRIC_COLUMNS}

    # Полный массив: каждая строка = эпоха, каждая колонка = метрика
    summary = pd.DataFrame(di)
    summary.insert(0, 'epoch', range(len(summary)))
    return meta_df, summary


def write_export_xlsx(filepath, params, env_results,
                      comparison_params=None, comparison_env=None):
    """Запись двух листов Excel (meta + exp_epoch) по указанному пути без диалога."""
    meta_df, summary = build_export_frames(params, env_results,
                                           comparison_params, comparison_env)
    with pd.ExcelWriter(filepath) as writer:
        meta_df.to_excel(writer, sheet_name='meta', index=False)
        summary.to_excel(writer, sheet_name='exp_epoch', index=False)


def run_param_sweep(base_params, param1, values1, param2, values2, run_fn, write_fn,
                    out_dir, progress_cb=None, stop_check=None):
    """Последовательный перебор всех комбинаций значений двух параметров.

    Для каждой комбинации: глубокая копия base_params с переопределёнными
    param1/param2 -> run_fn(params, label) -> имя файла эксперимента ->
    write_fn(путь, params, env, elapsed, num_proc). Возвращает список имён
    записанных файлов. stop_check() при True прерывает цикл между комбинациями;
    progress_cb(i, total, param1, v1, param2, v2) вызывается после каждой записи.
    """
    written = []
    total = len(values1) * len(values2)
    i = 0
    for v1 in values1:
        for v2 in values2:
            if stop_check is not None and stop_check():
                break
            i += 1
            params = copy.deepcopy(base_params)
            params[param1] = v1
            params[param2] = v2
            label = "Комбинация %d/%d" % (i, total)
            _mean, env, elapsed, num_proc = run_fn(params, label)
            fname = sweep_filename(param1, v1, param2, v2)
            write_fn(os.path.join(out_dir, fname), params, env, elapsed, num_proc)
            written.append(fname)
            if progress_cb is not None:
                progress_cb(i, total, param1, v1, param2, v2)
        if stop_check is not None and stop_check():
            break
    return written


# GUI
class TrafficModelGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Модель эволюции транспортного поведения")
        self.root.geometry("1400x900")
        
        # Окно на весь экран
        root.state('zoomed')
        
        # Цвета темы
        self.bg_color = "#f8f9fa"
        self.primary_color = "#2E86AB"
        self.secondary_color = "#A23B72"
        self.accent_color = "#F18F01"
        self.success_color = "#06A77D"
        
        # Цвет окна
        root.configure(bg=self.bg_color)
        
        self.params = self.get_default_params()
        
        self.results = None
        self.env_results = None
        self.comparison_params = None  # параметры сравнения
        self.comparison_results = None  # результаты сравнения
        
        self.current_chart_index = 0
        
        self.chart_names = [
            "Количество погибших",
            "Количество аварий",
            "Доля пристегнутых",
            "Доля превышающих скорость",
            "Доход правительства",
            "Доля строгих правительств",
            "Среднее время ожидания",
            "Среднее время ожидания без смертей",
            "Средний коэффициент обслуживания",
            "Средняя скорость",
            "Кол-во пьющих водителей",
            "Кол-во и доля пьющих в эпохе",
            "Кол-во и доля пьющих, остановленных полицией",
            "Число активных водителей в эпохе",
            "Кол-во водителей в тюрьме",
            "Кол-во водителей без лицензии",
        ]

        self.chart_keys = [
            'fatal', 'crashes', 'Belted_NotBelted', 'Speedings', 'earns',
            'Strict_Lite', 'averageWaiting', 'averageWaitingNoFat',
            'averageMaint', 'averageSpeed',
            'Drunk', 'DrunkShare', 'DrunkCheckedShare', 'ActivePopulation',
            'Jailed', 'NoLicense',
        ]

        self.selected_charts = [True] * 16
        
        self.main_frame = ttk.Frame(root)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.show_main_menu()
    
    def get_default_params(self):
        return {
            
            'N_runs': 10,
            'N_epochs': 300,
            'N_drivers': 1000,
            'crash_cost': 11356151.627336528,
            'fatal_belt_min': 0.6,
            'delta_maintenance': 0.01,
            'random_seed': '',  # пусто = случайный seed

            'fatal_coeff': 100,
            'crash_speed_coeff': 1.5,
            'decrease_coeff': 0.8,   # θ_d — время поездки при превышении
            'spendings_tolarance_coeff': 0.8,
            'fee_tolarance_coeff': 0.8,
            'chance_of_mutation': 0.05,
            'check_coeff': 60,

            'chance_crash_s': 0.0001477937316216022,
            'chance_fatal_s': 1.6e-05,
            'mu_police_time_s': 0.27,
            'sigma_police_time_s': 0.05,
            'check_duration_s': 0.25,
            'check_chance_s': 0.6,
            'traffic_spendings_s': 177722.7274509804,
            'fee_belt_min_s': 11700,
            'fee_belt_max_s': 23400,
            'fee_speeding_min_s': 9000,
            'fee_speeding_max_s': 126900,
            'maintance_coeff_s': 0.8,

            'chance_crash_l': 0.00258533743337677,
            'chance_fatal_l': 0.0002798860173578712,
            'mu_police_time_l': 0.27,
            'sigma_police_time_l': 0.05,
            'check_duration_l': 0.25,
            'check_chance_l': 0.4,
            'traffic_spendings_l': 5740.08575517796,
            'fee_belt_min_l': 1500,
            'fee_belt_max_l': 3000,
            'fee_speeding_min_l': 750,
            'fee_speeding_max_l': 2250,
            'maintance_coeff_l': 0.6,

            'speed_n_min': 70,
            'speed_n_max': 100,
            'speed_s_min': 110,
            'speed_s_max': 150,
            'exp_road_duration': 1,
            'sigma_road_duration': 0.1,
            'length': 100,

            'drunk_min': 0.1,
            'drunk_max': 0.9,
            'theta_dr': 0.5,          # θ_dr
            'fatal_drunk_coeff': 1.5, # f_dr
            'jail_coeff_decrease': 0.8,   # θ_j
            'licence_coeff_decrease': 0.8,  # θ_l
            'conditional_decision_decrease': 0.5,  # θ_con
            'cost_jail_per_epoch_m': 361532.77509439626,  # содержание заключённого (мягкое)
            'cost_jail_per_epoch_s': 5839841.518438178,   # содержание заключённого (строгое)
            'lite_g_chance': 0.05,
            'strict_g_chance': 0.05,
            'court_checkpoint': 'court_decision_artifacts_combined',
            'court_dataset': 'koap_vectors_combined/vectors.csv',
            'court_pool_size': 10000,
            **LEGAL_OVERRIDE_DEFAULTS,
        }
    
    def show_main_menu(self):
        self.clear_frame()
        
        container = ttk.Frame(self.main_frame)
        container.place(relx=0.5, rely=0.5, anchor=tk.CENTER)
        
        title_label = tk.Label(container, text="Модель эволюции\nтранспортного поведения", 
                              font=("Arial", 28, "bold"),
                              fg=self.primary_color,
                              bg=self.bg_color)
        title_label.pack(pady=30)
        
        subtitle_label = tk.Label(container, text="Графический интерфейс пользователя", 
                                 font=("Arial", 12),
                                 fg="#666",
                                 bg=self.bg_color)
        subtitle_label.pack(pady=(0, 40))
        
        btn_frame = ttk.Frame(container)
        btn_frame.pack(pady=10)
        
        style = ttk.Style()
        style.configure('Large.TButton', font=('Arial', 12), padding=15)
        
        start_btn = ttk.Button(btn_frame, text="Начать моделирование", 
                              command=self.start_simulation,
                              style='Large.TButton',
                              width=28)
        start_btn.pack(pady=8)
        
        settings_btn = ttk.Button(btn_frame, text="Настройки параметров", 
                                 command=self.show_settings,
                                 style='Large.TButton',
                                 width=28)
        settings_btn.pack(pady=8)
        
        exit_btn = ttk.Button(btn_frame, text="Выход", 
                             command=self.root.quit,
                             style='Large.TButton',
                             width=28)
        exit_btn.pack(pady=8)
    
    def show_settings(self):
        self.clear_frame()
        
        canvas = tk.Canvas(self.main_frame, bg=self.bg_color, highlightthickness=0)
        scrollbar_y = ttk.Scrollbar(self.main_frame, orient="vertical", command=canvas.yview)
        scrollbar_x = ttk.Scrollbar(self.main_frame, orient="horizontal", command=canvas.xview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar_y.set, xscrollcommand=scrollbar_x.set)
        
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        
        canvas.bind_all("<MouseWheel>", _on_mousewheel)
        
        title_label = tk.Label(scrollable_frame, text="Настройки параметров", 
                              font=("Arial", 20, "bold"),
                              fg=self.primary_color,
                              bg=self.bg_color)
        title_label.pack(pady=15)
        
        self.create_param_section(scrollable_frame, "Мета-параметры", [
            ('N_runs', 'int'),
            ('N_epochs', 'int'),
            ('N_drivers', 'int'),
            ('crash_cost', 'float'),
            ('fatal_belt_min', 'float'),
            ('delta_maintenance', 'float'),
            ('random_seed', 'str'),
        ])

        self.create_param_section(scrollable_frame, "Параметры водителей", [
            ('speed_n_min', 'float'),
            ('speed_n_max', 'float'),
            ('speed_s_min', 'float'),
            ('speed_s_max', 'float'),
            ('exp_road_duration', 'float'),
            ('sigma_road_duration', 'float'),
            ('length', 'float'),
            ('drunk_min', 'float'),
            ('drunk_max', 'float'),
        ])

        self.create_param_section(scrollable_frame, "Национальные коэффициенты", [
            ('fatal_coeff', 'float'),
            ('crash_speed_coeff', 'float'),
            ('decrease_coeff', 'float'),
            ('spendings_tolarance_coeff', 'float'),
            ('fee_tolarance_coeff', 'float'),
            ('chance_of_mutation', 'float'),
            ('check_coeff', 'float'),
            ('lite_g_chance', 'float'),
            ('strict_g_chance', 'float'),
            ('theta_dr', 'float'),
            ('fatal_drunk_coeff', 'float'),
            ('jail_coeff_decrease', 'float'),
            ('licence_coeff_decrease', 'float'),
            ('conditional_decision_decrease', 'float'),
        ])

        self.create_param_section(scrollable_frame, "Параметры строгого правительства", [
            ('chance_crash_s', 'float'),
            ('chance_fatal_s', 'float'),
            ('mu_police_time_s', 'float'),
            ('sigma_police_time_s', 'float'),
            ('check_duration_s', 'float'),
            ('check_chance_s', 'float'),
            ('traffic_spendings_s', 'float'),
            ('fee_belt_min_s', 'float'),
            ('fee_belt_max_s', 'float'),
            ('fee_speeding_min_s', 'float'),
            ('fee_speeding_max_s', 'float'),
            ('maintance_coeff_s', 'float'),
            ('cost_jail_per_epoch_s', 'float'),
        ])

        self.create_param_section(scrollable_frame, "Параметры мягкого правительства", [
            ('chance_crash_l', 'float'),
            ('chance_fatal_l', 'float'),
            ('mu_police_time_l', 'float'),
            ('sigma_police_time_l', 'float'),
            ('check_duration_l', 'float'),
            ('check_chance_l', 'float'),
            ('traffic_spendings_l', 'float'),
            ('fee_belt_min_l', 'float'),
            ('fee_belt_max_l', 'float'),
            ('fee_speeding_min_l', 'float'),
            ('fee_speeding_max_l', 'float'),
            ('maintance_coeff_l', 'float'),
            ('cost_jail_per_epoch_m', 'float'),
        ])

        # Параметры суда: путь к CSV датасета дел с кнопкой «Обзор…»
        court_frame = ttk.LabelFrame(scrollable_frame, text="Параметры суда", padding=12)
        court_frame.pack(fill=tk.X, padx=15, pady=8)
        row = ttk.Frame(court_frame)
        row.pack(fill=tk.X, pady=4)
        ttk.Label(row, text="Датасет дел (CSV):", anchor=tk.W).pack(side=tk.LEFT)
        self.dataset_entry = ttk.Entry(row, width=40)
        self.dataset_entry.insert(0, str(self.params.get('court_dataset', '')))
        self.dataset_entry.pack(side=tk.LEFT, padx=5)
        ttk.Button(row, text="Обзор…", command=self.browse_dataset).pack(side=tk.LEFT, padx=5)

        for key, label in [('court_pool_size', 'Размер пула вердиктов (реализация)'),
                           ('court_checkpoint', 'Checkpoint модели (реализация)')]:
            row = ttk.Frame(court_frame)
            row.pack(fill=tk.X, pady=4)
            ttk.Label(row, text=label + ":", anchor=tk.W).pack(side=tk.LEFT)
            entry = ttk.Entry(row, width=40)
            entry.insert(0, str(self.params.get(key, '')))
            entry.pack(side=tk.LEFT, padx=5)
            self.param_entries[key] = (entry, 'str' if key == 'court_checkpoint' else 'int')

        style = ttk.Style()
        style.configure('Medium.TButton', font=('Arial', 11), padding=10)

        self.create_regulation_section(scrollable_frame)
        self.create_chart_selection_section(scrollable_frame)
        self.create_sweep_section(scrollable_frame)

        btn_frame = ttk.Frame(scrollable_frame)
        btn_frame.pack(pady=20)

        save_btn = ttk.Button(btn_frame, text="Сохранить и вернуться", 
                             command=self.save_and_return,
                             style='Medium.TButton',
                             width=22)
        save_btn.pack(side=tk.LEFT, padx=8)
        
        reset_btn = ttk.Button(btn_frame, text="Сбросить по умолчанию", 
                              command=self.reset_to_defaults,
                              style='Medium.TButton',
                              width=22)
        reset_btn.pack(side=tk.LEFT, padx=8)
        
        # кнопки режима сравнения
        comp_btn_frame = ttk.Frame(scrollable_frame)
        comp_btn_frame.pack(pady=10)
        
        save_comp_btn = ttk.Button(comp_btn_frame, text="Сохранить как настройки сравнения", 
                                   command=self.save_comparison_params,
                                   style='Medium.TButton',
                                   width=30)
        save_comp_btn.pack(side=tk.LEFT, padx=8)
        
        clear_comp_btn = ttk.Button(comp_btn_frame, text="Очистить настройки сравнения", 
                                    command=self.clear_comparison_params,
                                    style='Medium.TButton',
                                    width=30)
        clear_comp_btn.pack(side=tk.LEFT, padx=8)
        
        cancel_btn = ttk.Button(btn_frame, text="Отмена", 
                               command=self.show_main_menu,
                               style='Medium.TButton',
                               width=22)
        cancel_btn.pack(side=tk.LEFT, padx=8)
        
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        scrollbar_y.pack(side=tk.RIGHT, fill=tk.Y)
        scrollbar_x.pack(side=tk.BOTTOM, fill=tk.X)

    def browse_dataset(self):
        """Выбор CSV-файла датасета дел (кнопка «Обзор…»)."""
        from tkinter import filedialog
        filename = filedialog.askopenfilename(
            title="Выберите CSV с делами (формат vectors.csv)",
            filetypes=[("CSV файл", "*.csv"), ("Все файлы", "*.*")],
            initialdir=PROJECT_ROOT)
        if filename:
            # относительный путь (стабилен при переносе проекта)
            try:
                rel = os.path.relpath(filename, PROJECT_ROOT)
                stored = rel if rel != '..' and not rel.startswith('..' + os.sep) else filename
            except ValueError:
                stored = filename
            self.dataset_entry.delete(0, tk.END)
            self.dataset_entry.insert(0, stored)
            self.params['court_dataset'] = stored

    def create_param_section(self, parent, title, params):
        group_frame = ttk.LabelFrame(parent, text=title, padding=12)
        group_frame.pack(fill=tk.X, padx=15, pady=8)
        

        if not hasattr(self, 'param_entries'):
            self.param_entries = {}
        
        unit_map = {
            'mu_police_time_s': 'час',
            'sigma_police_time_s': 'час',
            'mu_police_time_l': 'час',
            'sigma_police_time_l': 'час',
            'check_duration_s': 'час',
            'check_duration_l': 'час',
            'exp_road_duration': 'час',
            'sigma_road_duration': 'час',
            
            'length': 'км',
            
            'speed_n_min': 'км/ч',
            'speed_n_max': 'км/ч',
            'speed_s_min': 'км/ч',
            'speed_s_max': 'км/ч',
            
            'traffic_spendings_s': 'руб',
            'traffic_spendings_l': 'руб',
            'fee_belt_min_s': 'руб',
            'fee_belt_max_s': 'руб',
            'fee_belt_min_l': 'руб',
            'fee_belt_max_l': 'руб',
            'fee_speeding_min_s': 'руб',
            'fee_speeding_max_s': 'руб',
            'fee_speeding_min_l': 'руб',
            'fee_speeding_max_l': 'руб',
            'crash_cost': 'руб',
            
            'maintance_coeff_s': '',
            'maintance_coeff_l': '',
            'chance_fatal_s': '',
            'chance_fatal_l': '',
            'chance_crash_s': '',
            'chance_crash_l': '',
            'check_chance_s': '',
            'check_chance_l': '',
            'chance_of_mutation': '',
            'check_coeff': '',
            'fatal_coeff': '',
            'crash_speed_coeff': '',
            'spendings_tolarance_coeff': '',
            'fee_tolarance_coeff': '',
            'delta_maintenance': '',
            'fatal_belt_min': '',
            
            'N_runs': 'шт',
            'N_epochs': 'шт',
            'N_drivers': 'чел',
            'random_seed': '',
        }
        
        for idx, (param_name, param_type) in enumerate(params):
            row = idx // 2
            col = idx % 2

            frame = ttk.Frame(group_frame)
            frame.grid(row=row, column=col, sticky=tk.W, padx=8, pady=4)

            label_text = PARAM_LABELS.get(param_name, param_name)
            label = ttk.Label(frame, text=f"{label_text}:", anchor=tk.W)
            label.pack(side=tk.LEFT)
            
            entry = ttk.Entry(frame, width=12)
            entry.insert(0, str(self.params[param_name]))
            entry.pack(side=tk.LEFT, padx=5)
            
            unit = unit_map.get(param_name, '')
            if unit:
                unit_label = ttk.Label(frame, text=unit, 
                                      foreground="#666", font=("Arial", 9))
                unit_label.pack(side=tk.LEFT, padx=3)
            
            self.param_entries[param_name] = (entry, param_type)
    
    def create_chart_selection_section(self, parent):
        group_frame = ttk.LabelFrame(parent, text="Выбор графиков для отображения", padding=12)
        group_frame.pack(fill=tk.X, padx=15, pady=8)
        
        checkbox_frame = ttk.Frame(group_frame)
        checkbox_frame.pack(fill=tk.X, pady=8)
        
        self.chart_checkboxes = []
        
        for i, chart_name in enumerate(self.chart_names):
            row = i // 2
            col = i % 2
            
            var = tk.BooleanVar(value=True)
            checkbox = ttk.Checkbutton(
                checkbox_frame,
                text=chart_name,
                variable=var,
                width=38
            )
            checkbox.grid(row=row, column=col, sticky=tk.W, padx=10, pady=3)
            
            self.chart_checkboxes.append(var)
        
        btn_frame = ttk.Frame(group_frame)
        btn_frame.pack(fill=tk.X, pady=8)
        
        select_all_btn = ttk.Button(btn_frame, text="Выбрать все",
                                   command=self.select_all_charts,
                                   width=18)
        select_all_btn.pack(side=tk.LEFT, padx=8)
        
        deselect_all_btn = ttk.Button(btn_frame, text="✗ Снять все",
                                     command=self.deselect_all_charts,
                                     width=18)
        deselect_all_btn.pack(side=tk.LEFT, padx=8)

    def create_sweep_section(self, parent):
        """Параметрический сценарий: выбор двух параметров и их диапазонов."""
        group_frame = ttk.LabelFrame(parent, text="Параметрический сценарий", padding=12)
        group_frame.pack(fill=tk.X, padx=15, pady=8)

        self.sweep_p1_var = tk.StringVar(value=SWEEPABLE_PARAMS[0] if SWEEPABLE_PARAMS else '')
        self.sweep_p2_var = tk.StringVar(value=SWEEPABLE_PARAMS[1] if len(SWEEPABLE_PARAMS) > 1 else '')

        # строки выбора параметров
        row1 = ttk.Frame(group_frame)
        row1.pack(fill=tk.X, pady=3)
        ttk.Label(row1, text="Параметр 1:", width=12).pack(side=tk.LEFT)
        ttk.Combobox(row1, values=SWEEPABLE_PARAMS, textvariable=self.sweep_p1_var,
                     width=34, state='readonly').pack(side=tk.LEFT, padx=5)
        self.sweep_p1_desc = ttk.Label(row1, text="", foreground="#666", font=("Arial", 9))
        self.sweep_p1_desc.pack(side=tk.LEFT, padx=5)

        row2 = ttk.Frame(group_frame)
        row2.pack(fill=tk.X, pady=3)
        ttk.Label(row2, text="Параметр 2:", width=12).pack(side=tk.LEFT)
        ttk.Combobox(row2, values=SWEEPABLE_PARAMS, textvariable=self.sweep_p2_var,
                     width=34, state='readonly').pack(side=tk.LEFT, padx=5)
        self.sweep_p2_desc = ttk.Label(row2, text="", foreground="#666", font=("Arial", 9))
        self.sweep_p2_desc.pack(side=tk.LEFT, padx=5)

        # диапазоны: начало / конец / шаг
        self.sweep_p1_start = tk.StringVar(value='')
        self.sweep_p1_end = tk.StringVar(value='')
        self.sweep_p1_step = tk.StringVar(value='')
        self.sweep_p2_start = tk.StringVar(value='')
        self.sweep_p2_end = tk.StringVar(value='')
        self.sweep_p2_step = tk.StringVar(value='')

        def _range_row(prefix, start_var, end_var, step_var):
            row = ttk.Frame(group_frame)
            row.pack(fill=tk.X, pady=3)
            ttk.Label(row, text=prefix + ":", width=12).pack(side=tk.LEFT)
            for lbl, var in [("Начало", start_var), ("Конец", end_var), ("Шаг", step_var)]:
                ttk.Label(row, text=lbl).pack(side=tk.LEFT, padx=(6, 0))
                ttk.Entry(row, width=14, textvariable=var).pack(side=tk.LEFT, padx=2)

        _range_row("Диапазон 1", self.sweep_p1_start, self.sweep_p1_end, self.sweep_p1_step)
        _range_row("Диапазон 2", self.sweep_p2_start, self.sweep_p2_end, self.sweep_p2_step)

        # живой счётчик комбинаций
        self.sweep_total_label = ttk.Label(group_frame, text="Всего комбинаций: —")
        self.sweep_total_label.pack(anchor=tk.W, padx=12, pady=4)

        def _update_sweep_total(*_):
            try:
                v1 = generate_sweep_values(self.sweep_p1_start.get(),
                                           self.sweep_p1_end.get(),
                                           self.sweep_p1_step.get())
                v2 = generate_sweep_values(self.sweep_p2_start.get(),
                                           self.sweep_p2_end.get(),
                                           self.sweep_p2_step.get())
                self.sweep_total_label.config(text="Всего комбинаций: %d" % (len(v1) * len(v2)))
            except Exception:
                self.sweep_total_label.config(text="Всего комбинаций: —")

        for var in (self.sweep_p1_start, self.sweep_p1_end, self.sweep_p1_step,
                    self.sweep_p2_start, self.sweep_p2_end, self.sweep_p2_step):
            var.trace_add('write', _update_sweep_total)

        # подпись описания выбранного параметра
        def _bind_desc(var, desc_label):
            def _on_select(*_):
                desc_label.config(text=PARAM_LABELS.get(var.get(), ''))
            var.trace_add('write', _on_select)
            _on_select()

        _bind_desc(self.sweep_p1_var, self.sweep_p1_desc)
        _bind_desc(self.sweep_p2_var, self.sweep_p2_desc)

        # кнопка запуска
        run_frame = ttk.Frame(group_frame)
        run_frame.pack(fill=tk.X, pady=8)
        ttk.Button(run_frame, text="Запустить сценарий", command=self.start_sweep,
                   style='Medium.TButton', width=24).pack(side=tk.LEFT, padx=8)

    def create_regulation_section(self, parent):
        """Секция параметров регуляции ст. 12.8/264.1: два входа на параметр (мягкое/строгое)."""
        group_frame = ttk.LabelFrame(parent, text="Регуляция: параметры ст. 12.8 / 264.1", padding=12)
        group_frame.pack(fill=tk.X, padx=15, pady=8)
        if not hasattr(self, 'param_entries'):
            self.param_entries = {}
        cols = ttk.Frame(group_frame)
        cols.pack(fill=tk.X)
        lite_frame = ttk.Frame(cols)
        lite_frame.grid(row=0, column=0, sticky=tk.NW, padx=8)
        strict_frame = ttk.Frame(cols)
        strict_frame.grid(row=0, column=1, sticky=tk.NW, padx=8)
        ttk.Label(lite_frame, text="Мягкое правительство (Mild/Россия)", font=("Arial", 10, "bold")) \
            .grid(row=0, column=0, columnspan=2, sticky=tk.W, pady=(0, 4))
        ttk.Label(strict_frame, text="Строгое правительство (Strict/Москва)", font=("Arial", 10, "bold")) \
            .grid(row=0, column=0, columnspan=2, sticky=tk.W, pady=(0, 4))

        # базовые ключи ст. 12.8/264.1 (порядок = порядку LEGAL_OVERRIDE_DEFAULTS)
        legal_bases = [key[:-2] for key in LEGAL_OVERRIDE_DEFAULTS if key.endswith('_l')]
        for idx, base in enumerate(legal_bases):
            label_text = PARAM_LABELS[base + '_l']
            row = idx + 1
            ttk.Label(lite_frame, text=f"{label_text}:", anchor=tk.W) \
                .grid(row=row, column=0, sticky=tk.W, pady=2)
            e_l = ttk.Entry(lite_frame, width=12)
            e_l.insert(0, str(self.params[base + '_l']))
            e_l.grid(row=row, column=1, sticky=tk.W, padx=(4, 0), pady=2)
            self.param_entries[base + '_l'] = (e_l, 'float')
            ttk.Label(strict_frame, text=f"{label_text}:", anchor=tk.W) \
                .grid(row=row, column=0, sticky=tk.W, pady=2)
            e_s = ttk.Entry(strict_frame, width=12)
            e_s.insert(0, str(self.params[base + '_s']))
            e_s.grid(row=row, column=1, sticky=tk.W, padx=(4, 0), pady=2)
            self.param_entries[base + '_s'] = (e_s, 'float')

    def select_all_charts(self):
        for var in self.chart_checkboxes:
            var.set(True)
    
    def deselect_all_charts(self):
        for var in self.chart_checkboxes:
            var.set(False)

    def _collect_and_validate_params(self):
        """Прочитать и проверить все поля настроек (общий код для сохранения,
        сравнения и параметрического сценария).

        Возвращает dict параметров (включая court_dataset) без изменения
        self.params. Бросает ValueError с русским сообщением — вызывающий
        показывает messagebox.
        """
        params = {}
        for param_name, (entry, param_type) in self.param_entries.items():
            value_str = entry.get().strip()

            if not value_str:
                # random_seed может быть пустым
                if param_name == 'random_seed':
                    params[param_name] = ''
                    continue
                else:
                    raise ValueError(f"Параметр '{param_name}' не может быть пустым")

            if param_type == 'int':
                value = int(float(value_str))
            elif param_type == 'str':
                value = value_str
            else:
                value = float(value_str)

            # проверка отрицательных значений (только числа)
            if param_type != 'str' and value < 0 and param_name not in ['crash_cost']:
                raise ValueError(f"Значение параметра '{param_name}' не может быть отрицательным")

            params[param_name] = value

        if params.get('court_pool_size', 1) <= 0:
            raise ValueError("Размер пула вердиктов (court_pool_size) должен быть больше нуля")

        params['court_dataset'] = self.dataset_entry.get().strip() or params.get('court_dataset', '')
        return params

    def save_and_return(self):
        try:
            self.params = self._collect_and_validate_params()

            self.selected_charts = [var.get() for var in self.chart_checkboxes]

            if not any(self.selected_charts):
                raise ValueError("Выберите хотя бы один график для отображения")

            messagebox.showinfo("Успех", "Параметры сохранены!")
            self.show_main_menu()

        except ValueError as e:
            messagebox.showerror("Ошибка", f"Ошибка проверки параметров: {str(e)}")

    def reset_to_defaults(self):
        if messagebox.askyesno("Подтверждение", "Вы уверены, что хотите сбросить все параметры по умолчанию?"):
            self.params = self.get_default_params()
            self.selected_charts = [True] * 16
            self.comparison_params = None  # параметры сравнения
            self.comparison_results = None  # результаты сравнения
            self.show_settings()
    
    def save_comparison_params(self):
        """Сохранить текущие настройки как настройки сравнения"""
        try:
            # проверка параметров (без изменения self.params)
            self._collect_and_validate_params()

            # сохранить текущие параметры как настройки сравнения
            self.comparison_params = self.params.copy()

            messagebox.showinfo("Успех", "Настройки сравнения сохранены!\nТеперь моделирование будет выполняться дважды.")

        except ValueError as e:
            messagebox.showerror("Ошибка", f"Ошибка проверки параметров: {str(e)}")
    
    def clear_comparison_params(self):
        """Очистить настройки сравнения"""
        if self.comparison_params is None:
            messagebox.showinfo("Информация", "Настройки сравнения уже пусты")
            return
        
        if messagebox.askyesno("Подтверждение", "Вы уверены, что хотите очистить настройки сравнения?"):
            self.comparison_params = None
            self.comparison_results = None
            messagebox.showinfo("Успех", "Настройки сравнения очищены")
    
    def start_simulation(self):
        progress_window = tk.Toplevel(self.root)
        progress_window.title("Моделирование выполняется")
        progress_window.geometry("450x220")
        progress_window.transient(self.root)
        progress_window.grab_set()
        progress_window.configure(bg=self.bg_color)
        
        label = tk.Label(progress_window, text="Выполняется моделирование...", 
                        font=("Arial", 13, "bold"),
                        fg=self.primary_color,
                        bg=self.bg_color)
        label.pack(pady=25)
        
        progress = ttk.Progressbar(progress_window, mode='indeterminate', length=350)
        progress.pack(fill=tk.X, padx=30, pady=10)
        progress.start()
        
        status_label = tk.Label(progress_window, text="Подготовка...", 
                               font=("Arial", 10),
                               fg="#666",
                               bg=self.bg_color)
        status_label.pack(pady=10)
        
        self.root.after(100, lambda: self.run_simulation_in_background(progress_window, status_label))
    
    def run_simulation_in_background(self, progress_window, status_label):
        """Запуск симуляции в фоновом потоке (GUI не блокируется)"""
        def simulation_worker():
            """Рабочая функция фонового потока"""
            try:
                is_comparison_mode = self.comparison_params is not None
                
                # seed: если пусто — случайный
                p = self.params
                seed_str = str(p.get('random_seed', '')).strip()
                
                if seed_str == '':
                    used_seed = np.random.randint(0, 2147483647)
                else:
                    try:
                        used_seed = int(seed_str)
                    except ValueError:
                        used_seed = np.random.randint(0, 2147483647)
                
                np.random.seed(used_seed)
                random.seed(used_seed)
                
                # первое моделирование
                self.root.after(0, lambda: status_label.config(text="Первое моделирование: запуск..."))
                results1, env1, time1, num_proc1 = self._run_single_simulation(p, status_label, "Первое")
                
                if is_comparison_mode:
                    self.root.after(0, lambda: status_label.config(text="Режим сравнения: запуск второго моделирования..."))
                    
                    comp_p = self.comparison_params
                    comp_seed_str = str(comp_p.get('random_seed', '')).strip()
                    
                    if comp_seed_str == '':
                        used_comp_seed = np.random.randint(0, 2147483647)
                    else:
                        try:
                            used_comp_seed = int(comp_seed_str)
                        except ValueError:
                            used_comp_seed = np.random.randint(0, 2147483647)
                    
                    np.random.seed(used_comp_seed)
                    random.seed(used_comp_seed)
                    
                    results2, env2, time2, num_proc2 = self._run_single_simulation(comp_p, status_label, "Второе")
                    
                    # результат в временные переменные (потокобезопасно)
                    self._sim_results = {
                        'results1': results1, 'env1': env1, 'time1': time1, 'num_proc1': num_proc1, 'seed1': used_seed,
                        'results2': results2, 'env2': env2, 'time2': time2, 'num_proc2': num_proc2, 'seed2': used_comp_seed,
                        'is_comparison': True
                    }
                else:
                    self._sim_results = {
                        'results1': results1, 'env1': env1, 'time1': time1, 'num_proc1': num_proc1, 'seed1': used_seed,
                        'is_comparison': False
                    }
                
                # обновление GUI в главном потоке
                self.root.after(10, self._on_simulation_complete)
                
            except Exception as e:
                self._sim_error = str(e)
                self.root.after(10, lambda: self._on_simulation_error(progress_window))
        
        # очистка старых результатов
        self._sim_results = None
        self._sim_error = None
        
        # запуск фонового потока
        thread = threading.Thread(target=simulation_worker, daemon=True)
        thread.start()
    
    def _on_simulation_complete(self):
        """Обновление GUI в главном потоке после симуляции"""
        # закрыть окна прогресса
        for w in self.root.winfo_children():
            if isinstance(w, tk.Toplevel):
                w.destroy()
        
        if self._sim_results is None:
            return
        
        r = self._sim_results
        
        # освобождение старых данных
        self.results = None
        self.env_results = None
        self.comparison_results = None
        gc.collect()
        
        self.results = r['results1']
        self.env_results = r['env1']
        self.simulation_time = r['time1']
        self.num_processes = r['num_proc1']
        self.used_seed = r['seed1']
        
        if r['is_comparison']:
            self.comparison_results = {
                'results': r['results2'],
                'env': r['env2'],
                'time': r['time2'],
                'num_processes': r['num_proc2'],
                'seed': r['seed2']
            }
        
        # очистка временных данных
        self._sim_results = None
        
        self.show_results()
    
    def _on_simulation_error(self, progress_window):
        """Показ ошибки в главном потоке"""
        try:
            progress_window.destroy()
        except Exception:
            pass
        messagebox.showerror("Ошибка", f"Ошибка при выполнении моделирования: {self._sim_error}")
    
    def _run_single_simulation(self, params, status_label, label, court_pool=None):
        """Запуск одной симуляции и возврат результата.

        При переданном court_pool (параметрический сценарий) пул вердиктов не
        пересоздаётся — экономия на построении пула для каждой комбинации.
        """
        p = params

        # Пул вердиктов (мягкий + строгий) один раз на симуляцию (кэшируется в .pkl)
        if court_pool is None:
            self.root.after(0, lambda: status_label.config(text="Суд: построение пула вердиктов..."))
            court = CourtSystem(p['court_checkpoint'], p['court_dataset'], int(p['court_pool_size']))
            court_pool = build_court_pool(court, p)

        params_tuple = (
            0, p['N_runs'], p['N_epochs'], p['N_drivers'],
            p['fatal_belt_min'],
            p['fatal_coeff'], p['crash_speed_coeff'], p['crash_cost'],
            p['spendings_tolarance_coeff'],
            p['chance_of_mutation'], p['delta_maintenance'],
            p['fee_tolarance_coeff'], p['check_coeff'], p['decrease_coeff'],
            p['chance_crash_s'], p['mu_police_time_s'], p['sigma_police_time_s'],
            p['check_duration_s'], p['check_chance_s'],
            p['traffic_spendings_s'], p['fee_belt_min_s'], p['fee_belt_max_s'],
            p['fee_speeding_min_s'], p['fee_speeding_max_s'], p['maintance_coeff_s'],
            p['chance_crash_l'], p['mu_police_time_l'], p['sigma_police_time_l'],
            p['check_duration_l'], p['check_chance_l'],
            p['traffic_spendings_l'], p['fee_belt_min_l'], p['fee_belt_max_l'],
            p['fee_speeding_min_l'], p['fee_speeding_max_l'], p['maintance_coeff_l'],
            p['speed_n_min'], p['speed_n_max'], p['speed_s_min'], p['speed_s_max'],
            p['exp_road_duration'], p['sigma_road_duration'], p['length'],
            p['chance_fatal_l'], p['chance_fatal_s'],
            p['drunk_min'], p['drunk_max'],
            p['theta_dr'], p['fatal_drunk_coeff'], p['jail_coeff_decrease'], p['licence_coeff_decrease'], p['conditional_decision_decrease'],
            p['cost_jail_per_epoch_m'], p['cost_jail_per_epoch_s'],
            p['lite_g_chance'], p['strict_g_chance'],
            court_pool,
        )

        params_list = []
        for k in range(p['N_runs']):
            sim_params = list(params_tuple)
            sim_params[0] = k
            params_list.append(tuple(sim_params))
        
        status_label_text = f"{label} моделирование: запуск {p['N_runs']} процессов..."
        self.root.after(0, lambda: status_label.config(text=status_label_text))
        
        start_time = time.time()
        # Кап воркеров: каждый — отдельный процесс numpy; 10 процессов при N_drivers=10000
        # исчерпывают страничный файл (WinError 1455). Кап 4.
        num_processes = min(cpu_count(), p['N_runs'], 4)

        with Pool(processes=num_processes) as pool:
            results = pool.map(run_single_experiment, params_list)
        
        elapsed_time = time.time() - start_time
        
        status_label_text2 = f"{label} моделирование: объединение результатов..."
        self.root.after(0, lambda: status_label.config(text=status_label_text2))
        
        accumulated = add_arr_parallel(results)
        mean_results = add_mean_parallel(accumulated, p['N_runs'])
        
        env = Env(p['N_drivers'], p['fatal_belt_min'],
                 p['fatal_coeff'], p['crash_speed_coeff'],
                 p['crash_cost'], p['spendings_tolarance_coeff'],
                 p['chance_of_mutation'], p['delta_maintenance'],
                 p['fee_tolarance_coeff'], p['check_coeff'])
        
        env.fatal = mean_results['fatal']
        env.crashes = mean_results['crashes']
        env.Belted_NotBelted = mean_results['Belted_NotBelted']
        env.Speedings = mean_results['Speedings']
        env.earns = mean_results['earns']
        env.Strict_Lite = mean_results['Strict_Lite']
        env.averageWaiting = mean_results['averageWaiting']
        env.averageMaint = mean_results['averageMaint']
        env.averageSpeed = mean_results['averageSpeed']
        env.amountOfFatal = mean_results['amountOfFatal']
        env.averageWaitingNoFat = mean_results['averageWaitingNoFat']
        env.epoch = p['N_epochs']
        env.Drunk = mean_results['Drunk']
        env.DrunkChecked = mean_results['DrunkChecked']
        env.ActivePopulation = mean_results['ActivePopulation']
        env.Jailed = mean_results['Jailed']
        env.NoLicense = mean_results['NoLicense']

        return mean_results, env, elapsed_time, num_processes

    def start_sweep(self):
        """Запуск параметрического сценария (перебор двух параметров)."""
        try:
            base_params = self._collect_and_validate_params()
        except ValueError as e:
            messagebox.showerror("Ошибка", f"Ошибка проверки параметров: {str(e)}")
            return

        try:
            p1 = self.sweep_p1_var.get().strip()
            p2 = self.sweep_p2_var.get().strip()
            if not p1 or not p2:
                raise ValueError("Выберите оба параметра для сценария")
            if p1 == p2:
                raise ValueError("Выберите два разных параметра")

            values1 = generate_sweep_values(self.sweep_p1_start.get(),
                                            self.sweep_p1_end.get(),
                                            self.sweep_p1_step.get())
            values2 = generate_sweep_values(self.sweep_p2_start.get(),
                                            self.sweep_p2_end.get(),
                                            self.sweep_p2_step.get())
            if not values1 or not values2:
                raise ValueError("Диапазон не содержит значений")

            total = len(values1) * len(values2)
            if total <= 0:
                raise ValueError("Нет комбинаций для запуска")
        except ValueError as e:
            messagebox.showerror("Ошибка", f"Ошибка параметров сценария: {str(e)}")
            return

        msg = (f"Параметр 1: {p1} → {len(values1)} значений\n"
               f"Параметр 2: {p2} → {len(values2)} значений\n"
               f"Всего комбинаций: {total}\n"
               f"Файлы будут сохранены в:\n{PROJECT_ROOT}\n\nЗапустить?")
        if not messagebox.askyesno("Подтверждение", msg):
            return

        # окно прогресса
        progress_window = tk.Toplevel(self.root)
        progress_window.title("Параметрический сценарий выполняется")
        progress_window.geometry("540x250")
        progress_window.transient(self.root)
        progress_window.grab_set()
        progress_window.configure(bg=self.bg_color)

        label = tk.Label(progress_window, text="Выполняется параметрический сценарий...",
                         font=("Arial", 13, "bold"), fg=self.primary_color, bg=self.bg_color)
        label.pack(pady=18)

        progress = ttk.Progressbar(progress_window, mode='determinate', maximum=total, length=440)
        progress.pack(fill=tk.X, padx=30, pady=8)

        status_label = tk.Label(progress_window, text="Подготовка...", font=("Arial", 10),
                                fg="#666", bg=self.bg_color)
        status_label.pack(pady=8)

        ttk.Button(progress_window, text="Отмена", width=18,
                   command=self._cancel_sweep).pack(pady=6)

        self._sweep_stop = False
        self.root.after(100, lambda: self._run_sweep_in_background(
            progress_window, status_label, progress,
            base_params, p1, values1, p2, values2))

    def _cancel_sweep(self):
        """Запросить остановку сценария (проверяется между комбинациями)."""
        self._sweep_stop = True

    def _run_sweep_in_background(self, progress_window, status_label, progress,
                                 base_params, p1, values1, p2, values2):
        """Запуск параметрического сценария в фоновом потоке (GUI не блокируется)."""
        def sweep_worker():
            try:
                # seed: если пусто — случайный
                seed_str = str(base_params.get('random_seed', '')).strip()
                if seed_str == '':
                    used_seed = np.random.randint(0, 2147483647)
                else:
                    try:
                        used_seed = int(seed_str)
                    except ValueError:
                        used_seed = np.random.randint(0, 2147483647)
                np.random.seed(used_seed)
                random.seed(used_seed)

                # Пул вердиктов строим один раз, если сценарий не меняет параметры
                # правовой регуляции (иначе перестраиваем на каждой комбинации).
                legal_keys = set(LEGAL_OVERRIDE_DEFAULTS)
                rebuild_pool = (p1 in legal_keys) or (p2 in legal_keys)
                prebuilt_pool = None
                if not rebuild_pool:
                    self.root.after(0, lambda: status_label.config(text="Суд: построение пула вердиктов..."))
                    court = CourtSystem(base_params['court_checkpoint'],
                                        base_params['court_dataset'],
                                        int(base_params['court_pool_size']))
                    prebuilt_pool = build_court_pool(court, base_params)

                start_time = time.time()

                def run_fn(params, label):
                    pool = None if rebuild_pool else prebuilt_pool
                    return self._run_single_simulation(params, status_label, label, court_pool=pool)

                def write_fn(fpath, params, env, elapsed, num_proc):
                    write_export_xlsx(fpath, params, env)

                def progress_cb(i, n, param1, v1, param2, v2):
                    self.root.after(0, lambda: progress.config(value=i))
                    self.root.after(0, lambda: status_label.config(
                        text="Комбинация %d/%d: %s=%s, %s=%s"
                             % (i, n, param1, format_param_value(v1),
                                param2, format_param_value(v2))))

                written = run_param_sweep(base_params, p1, values1, p2, values2,
                                          run_fn, write_fn, PROJECT_ROOT,
                                          progress_cb=progress_cb,
                                          stop_check=lambda: self._sweep_stop)

                elapsed_total = time.time() - start_time
                self._sweep_done = (written, elapsed_total, not self._sweep_stop)
            except Exception as e:
                self._sweep_error = str(e)
            self.root.after(10, lambda: self._on_sweep_complete(progress_window))

        self._sweep_error = None
        self._sweep_done = None
        thread = threading.Thread(target=sweep_worker, daemon=True)
        thread.start()

    def _on_sweep_complete(self, progress_window):
        """Обновление GUI в главном потоке после сценария."""
        try:
            progress_window.destroy()
        except Exception:
            pass

        if self._sweep_error is not None:
            messagebox.showerror("Ошибка", f"Ошибка сценария: {self._sweep_error}")
            self._sweep_error = None
            return

        written, elapsed_total, completed = self._sweep_done
        self._sweep_done = None
        if completed:
            messagebox.showinfo("Успех",
                                f"Сценарий выполнен: {len(written)} файлов\n"
                                f"Время: {elapsed_total:.1f} с\nПапка: {PROJECT_ROOT}")
        else:
            messagebox.showinfo("Отменено",
                                f"Сценарий остановлен. Экспортировано {len(written)} "
                                f"файлов.\nПапка: {PROJECT_ROOT}")

    def show_results(self):
        self.clear_frame()
        
        chart_frame = ttk.Frame(self.main_frame)
        chart_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.fig = Figure(figsize=(11, 6.5), dpi=100, facecolor=self.bg_color)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor('#ffffff')
        
        self.canvas = FigureCanvasTkAgg(self.fig, chart_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        self.toolbar = NavigationToolbar2Tk(self.canvas, chart_frame)
        self.toolbar.update()
        
        nav_frame = ttk.Frame(chart_frame)
        nav_frame.pack(pady=12)
        
        style = ttk.Style()
        style.configure('Nav.TButton', font=('Arial', 11), padding=8)
        
        prev_btn = ttk.Button(nav_frame, text="← Предыдущая", 
                             command=self.prev_chart,
                             style='Nav.TButton',
                             width=16)
        prev_btn.pack(side=tk.LEFT, padx=8)
        
        self.page_label = tk.Label(nav_frame, text="", 
                                  font=("Arial", 11, "bold"),
                                  fg=self.primary_color,
                                  bg=self.bg_color)
        self.page_label.pack(side=tk.LEFT, padx=15)
        
        next_btn = ttk.Button(nav_frame, text="Следующая →",
                             command=self.next_chart,
                             style='Nav.TButton',
                             width=16)
        next_btn.pack(side=tk.LEFT, padx=8)

        action_frame = ttk.Frame(chart_frame)
        action_frame.pack(pady=(0, 12))

        export_chart_btn = ttk.Button(action_frame, text="Экспорт графика",
                                     command=self.export_chart,
                                     style='Nav.TButton',
                                     width=18)
        export_chart_btn.pack(side=tk.LEFT, padx=6)

        export_data_btn = ttk.Button(action_frame, text="Экспорт данных",
                                    command=self.export_data,
                                    style='Nav.TButton',
                                    width=18)
        export_data_btn.pack(side=tk.LEFT, padx=6)

        back_btn = ttk.Button(action_frame, text="Вернуться в меню",
                             command=self.show_main_menu,
                             style='Nav.TButton',
                             width=18)
        back_btn.pack(side=tk.LEFT, padx=6)

        info_frame = ttk.LabelFrame(self.main_frame, text="Статистика результатов", padding=12)
        info_frame.pack(fill=tk.X, padx=10, pady=8)
        
        # режим сравнения?
        is_comparison_mode = self.comparison_results is not None
        
        if is_comparison_mode:
            perf_text = f"РЕЖИМ СРАВНЕНИЯ"
            perf_label = tk.Label(info_frame, text=perf_text, 
                                 font=("Arial", 12, "bold"),
                                 fg="#F18F01",
                                 bg=self.bg_color)
            perf_label.pack(anchor=tk.W, pady=5)
            
            perf_text1 = f"Первое моделирование: {self.simulation_time:.2f} сек  |  "
            perf_text1 += f"Процессы: {self.num_processes}  |  "
            perf_text1 += f"Эксперименты: {self.params['N_runs']}  |  "
            perf_text1 += f"Эпохи: {self.params['N_epochs']}"
            
            perf_label1 = tk.Label(info_frame, text=perf_text1, 
                                  font=("Arial", 10),
                                  fg="#333",
                                  bg=self.bg_color)
            perf_label1.pack(anchor=tk.W, pady=3)
            
            comp_time = self.comparison_results['time']
            comp_procs = self.comparison_results['num_processes']
            comp_seed = self.comparison_results['seed']
            
            perf_text2 = f"Второе моделирование: {comp_time:.2f} сек  |  "
            perf_text2 += f"Процессы: {comp_procs}  |  "
            perf_text2 += f"Эксперименты: {self.comparison_params['N_runs']}  |  "
            perf_text2 += f"Эпохи: {self.comparison_params['N_epochs']}"
            
            perf_label2 = tk.Label(info_frame, text=perf_text2, 
                                  font=("Arial", 10),
                                  fg="#333",
                                  bg=self.bg_color)
            perf_label2.pack(anchor=tk.W, pady=3)
            
            seed_text = f"Seed 1: {self.used_seed}  |  Seed 2: {comp_seed}"
            seed_label = tk.Label(info_frame, text=seed_text, 
                                 font=("Arial", 10, "bold"),
                                 fg=self.primary_color,
                                 bg=self.bg_color)
            seed_label.pack(anchor=tk.W, pady=3)
        else:
            perf_text = f"Время: {self.simulation_time:.2f} сек  |  "
            perf_text += f"Процессы: {self.num_processes}  |  "
            perf_text += f"Эксперименты: {self.params['N_runs']}  |  "
            perf_text += f"Эпохи: {self.params['N_epochs']}"
            
            perf_label = tk.Label(info_frame, text=perf_text, 
                                 font=("Arial", 10),
                                 fg="#333",
                                 bg=self.bg_color)
            perf_label.pack(anchor=tk.W, pady=3)
            
            # показать использованный seed
            seed_text = f"Seed: {self.used_seed}"
            seed_label = tk.Label(info_frame, text=seed_text, 
                                 font=("Arial", 10, "bold"),
                                 fg=self.primary_color,
                                 bg=self.bg_color)
            seed_label.pack(anchor=tk.W, pady=3)
        
        # статистика превышения и ремней
        if is_comparison_mode:
            # режим сравнения: статистика двух прогонов
            u_speeding_1 = np.array(self.env_results.Speedings)
            u_belted_1 = np.array(self.env_results.Belted_NotBelted)
            
            comp_env = self.comparison_results['env']
            u_speeding_2 = np.array(comp_env.Speedings)
            u_belted_2 = np.array(comp_env.Belted_NotBelted)
            u_drunk_1 = np.array(self.env_results.Drunk) / float(self.params['N_drivers'])
            u_drunk_2 = np.array(comp_env.Drunk) / float(self.comparison_params['N_drivers'])
            
            stats_title = tk.Label(info_frame, text="Статистика поведения водителей:", 
                                  font=("Arial", 11, "bold"),
                                  fg="#333",
                                  bg=self.bg_color)
            stats_title.pack(anchor=tk.W, pady=5)
            
            stats_text_1 = f"Первое моделирование:\n"
            stats_text_1 += f"   Превышение скорости: {u_speeding_1.mean():.2%} ± {u_speeding_1.std():.2%}  |  "
            stats_text_1 += f"Пристегнутые: {u_belted_1.mean():.2%} ± {u_belted_1.std():.2%}  |  "
            stats_text_1 += f"Пьяные: {u_drunk_1.mean():.2%} ± {u_drunk_1.std():.2%}"
            
            stats_label_1 = tk.Label(info_frame, text=stats_text_1,
                                    font=("Arial", 10),
                                    fg="#333",
                                    bg=self.bg_color,
                                    justify=tk.LEFT)
            stats_label_1.pack(anchor=tk.W, pady=3)

            stats_text_2 = f"Второе моделирование:\n"
            stats_text_2 += f"   Превышение скорости: {u_speeding_2.mean():.2%} ± {u_speeding_2.std():.2%}  |  "
            stats_text_2 += f"Пристегнутые: {u_belted_2.mean():.2%} ± {u_belted_2.std():.2%}  |  "
            stats_text_2 += f"Пьяные: {u_drunk_2.mean():.2%} ± {u_drunk_2.std():.2%}"

            stats_label_2 = tk.Label(info_frame, text=stats_text_2,
                                    font=("Arial", 10),
                                    fg="#333",
                                    bg=self.bg_color,
                                    justify=tk.LEFT)
            stats_label_2.pack(anchor=tk.W, pady=3)

            # разница
            speeding_diff = (u_speeding_2.mean() - u_speeding_1.mean()) * 100
            belted_diff = (u_belted_2.mean() - u_belted_1.mean()) * 100
            drunk_diff = (u_drunk_2.mean() - u_drunk_1.mean()) * 100

            diff_text = f"Разница (2-1):\n"
            diff_text += f"   Превышение скорости: {speeding_diff:+.2f}%  |  "
            diff_text += f"Пристегнутые: {belted_diff:+.2f}%  |  "
            diff_text += f"Пьяные: {drunk_diff:+.2f}%"
            
            diff_label = tk.Label(info_frame, text=diff_text,
                                 font=("Arial", 10, "bold"),
                                 fg=self.accent_color,
                                 bg=self.bg_color,
                                 justify=tk.LEFT)
            diff_label.pack(anchor=tk.W, pady=5)
        else:
            # обычный режим: статистика одного прогона (три КПИ раздела 2 в одной строке)
            u_speeding = np.array(self.env_results.Speedings)
            u_belted = np.array(self.env_results.Belted_NotBelted)
            u_drunk = np.array(self.env_results.Drunk) / float(self.params['N_drivers'])

            stats_text = f"Превышение скорости: {u_speeding.mean():.2%} ± {u_speeding.std():.2%}  |  "
            stats_text += f"Пристегнутые: {u_belted.mean():.2%} ± {u_belted.std():.2%}  |  "
            stats_text += f"Пьяные: {u_drunk.mean():.2%} ± {u_drunk.std():.2%}"

            stats_label = tk.Label(info_frame, text=stats_text,
                                  font=("Arial", 10),
                                  fg="#333",
                                  bg=self.bg_color)
            stats_label.pack(anchor=tk.W, pady=3)

        self.visible_chart_indices = [i for i, selected in enumerate(self.selected_charts) if selected]
        
        if not self.visible_chart_indices:
            self.visible_chart_indices = list(range(len(self.chart_names)))

        self.current_chart_index = 0
        self.update_chart()
    
    def update_chart(self):
        if self.env_results is None:
            return
        
        self.ax.clear()
        if getattr(self, '_ax2', None) is not None:
            self._ax2.remove()
            self._ax2 = None

        if not hasattr(self, 'visible_chart_indices'):
            self.visible_chart_indices = list(range(len(self.chart_names)))

        actual_index = self.visible_chart_indices[self.current_chart_index]

        def _share_of(count, pop):
            c = np.asarray(count, dtype=float)
            p = np.asarray(pop, dtype=float)
            return np.where(p > 0, c / p, 0.0)

        # data_map: (серия, заголовок, доля_или_None); «кол-во и доля» — обе серии (доля на правой оси)
        data_map = {
            0: (self.env_results.fatal, "Количество погибших", None),
            1: (self.env_results.crashes, "Количество аварий", None),
            2: (self.env_results.Belted_NotBelted, "Доля пристегнутых", None),
            3: (self.env_results.Speedings, "Доля превышающих скорость", None),
            4: (self.env_results.earns, "Доход правительства", None),
            5: (self.env_results.Strict_Lite, "Доля строгих правительств", None),
            6: (self.env_results.averageWaiting, "Среднее время ожидания", None),
            7: (self.env_results.averageWaitingNoFat, "Среднее время ожидания без смертей", None),
            8: (self.env_results.averageMaint, "Средний коэффициент обслуживания", None),
            9: (self.env_results.averageSpeed, "Средняя скорость", None),
            10: (self.env_results.Drunk, "Кол-во пьющих водителей", None),
            11: (self.env_results.Drunk, "Кол-во и доля пьющих в эпохе",
                 _share_of(self.env_results.Drunk, self.env_results.ActivePopulation)),
            12: (self.env_results.DrunkChecked, "Кол-во и доля пьющих, остановленных полицией",
                 _share_of(self.env_results.DrunkChecked, self.env_results.ActivePopulation)),
            13: (self.env_results.ActivePopulation, "Число активных водителей в эпохе", None),
            14: (self.env_results.Jailed, "Кол-во водителей в тюрьме", None),
            15: (self.env_results.NoLicense, "Кол-во водителей без лицензии", None),
        }

        if actual_index in data_map:
            data, title, share = data_map[actual_index]
            epochs = list(range(len(data)))
            is_comparison_mode = self.comparison_results is not None
            is_dual = share is not None

            if is_dual:
                # правая ось для доли (0..1)
                ax2 = self.ax.twinx()
                self._ax2 = ax2
                ax2.set_ylabel('Доля', fontsize=13, fontweight='bold')
                ax2.set_ylim(0, 1)

            if is_comparison_mode:
                comp_env = self.comparison_results['env']
                comp_data_map = {
                    0: comp_env.fatal,
                    1: comp_env.crashes,
                    2: comp_env.Belted_NotBelted,
                    3: comp_env.Speedings,
                    4: comp_env.earns,
                    5: comp_env.Strict_Lite,
                    6: comp_env.averageWaiting,
                    7: comp_env.averageWaitingNoFat,
                    8: comp_env.averageMaint,
                    9: comp_env.averageSpeed,
                    10: comp_env.Drunk,
                    11: comp_env.Drunk,
                    12: comp_env.DrunkChecked,
                    13: comp_env.ActivePopulation,
                    14: comp_env.Jailed,
                    15: comp_env.NoLicense,
                }
                comp_share_map = {
                    11: _share_of(comp_env.Drunk, comp_env.ActivePopulation),
                    12: _share_of(comp_env.DrunkChecked, comp_env.ActivePopulation),
                }

                if actual_index in comp_data_map:
                    comp_data = comp_data_map[actual_index]
                    comp_seed = self.comparison_results["seed"]
                    if is_dual:
                        comp_share = comp_share_map[actual_index]
                        self.ax.plot(epochs, data, linewidth=2.5, color='#2E86AB',
                                    marker='o', markersize=3, alpha=0.8,
                                    label=f'Кол-во (Seed: {self.used_seed})')
                        self.ax.plot(epochs, comp_data, linewidth=2.5, color='#F18F01',
                                    marker='s', markersize=3, alpha=0.8,
                                    label=f'Кол-во (Seed: {comp_seed})')
                        ax2.plot(epochs, share, linewidth=1.8, color='#2E86AB',
                                linestyle='--', marker='o', markersize=2.5, alpha=0.9,
                                label=f'Доля (Seed: {self.used_seed})')
                        ax2.plot(epochs, comp_share, linewidth=1.8, color='#F18F01',
                                linestyle='--', marker='s', markersize=2.5, alpha=0.9,
                                label=f'Доля (Seed: {comp_seed})')
                    else:
                        self.ax.plot(epochs, data, linewidth=2.5, color='#2E86AB',
                                    marker='o', markersize=3, alpha=0.8,
                                    label=f'Текущие настройки (Seed: {self.used_seed})')
                        self.ax.plot(epochs, comp_data, linewidth=2.5, color='#F18F01',
                                    marker='s', markersize=3, alpha=0.8,
                                    label=f'Настройки сравнения (Seed: {comp_seed})')
                    self.ax.legend(loc='best', fontsize=10, framealpha=0.9)
            else:
                if is_dual:
                    self.ax.plot(epochs, data, linewidth=2.5, color=self.primary_color,
                                marker='o', markersize=3, alpha=0.8, label='Кол-во')
                    ax2.plot(epochs, share, linewidth=1.8, color=self.accent_color,
                            linestyle='--', marker='o', markersize=2.5, alpha=0.9, label='Доля')
                    self.ax.legend(loc='best', fontsize=10, framealpha=0.9)
                else:
                    self.ax.plot(epochs, data, linewidth=2.5, color=self.primary_color,
                                marker='o', markersize=3, alpha=0.7)

            self.ax.set_title(title, fontsize=15, fontweight='bold', pad=15)
            self.ax.set_xlabel('Эпоха', fontsize=13, fontweight='bold')
            self.ax.set_ylabel(title, fontsize=13, fontweight='bold')
            self.ax.grid(True, alpha=0.3, linestyle='--')
            self.ax.tick_params(labelsize=11)

            total_visible = len(self.visible_chart_indices)
            self.page_label.config(text=f"{self.current_chart_index + 1} / {total_visible}")

        self.canvas.draw()
    
    def prev_chart(self):
        if self.current_chart_index > 0:
            self.current_chart_index -= 1
            self.update_chart()
    
    def next_chart(self):
        if self.current_chart_index < len(self.visible_chart_indices) - 1:
            self.current_chart_index += 1
            self.update_chart()
    
    def export_chart(self):
        file_types = [
            ("PNG изображение", "*.png"),
            ("PDF документ", "*.pdf"),
            ("SVG векторная графика", "*.svg"),
        ]
        
        filename = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=file_types,
            title="Экспорт графика"
        )
        
        if filename:
            try:
                self.fig.savefig(filename, dpi=300, bbox_inches='tight', facecolor=self.fig.get_facecolor())
                messagebox.showinfo("Успех", f"График экспортирован в:\n{filename}")
            except Exception as e:
                messagebox.showerror("Ошибка", f"Ошибка экспорта: {str(e)}")
    
    def export_data(self):
        # Формат экспорта: два листа — «meta» (параметры: первая строка — имена
        # в системе, вторая — значения при инициализации модели) и «exp_epoch»
        # (сводка: средние значения показателей после 10 эпох). CSV — тот же
        # набор секций в одном плоском файле.
        file_types = [
            ("Excel файл", "*.xlsx"),
            ("CSV файл", "*.csv"),
        ]

        filename = filedialog.asksaveasfilename(
            defaultextension=".xlsx",
            filetypes=file_types,
            title="Экспорт данных"
        )

        if filename:
            try:
                comp_params = None
                comp_env = None
                if self.comparison_results is not None:
                    comp_params = self.comparison_params
                    comp_env = self.comparison_results['env']

                meta_df, summary = build_export_frames(
                    self.params, self.env_results, comp_params, comp_env)

                if filename.endswith('.xlsx'):
                    with pd.ExcelWriter(filename) as writer:
                        meta_df.to_excel(writer, sheet_name='meta', index=False)
                        summary.to_excel(writer, sheet_name='exp_epoch', index=False)
                else:
                    with open(filename, 'w', encoding='utf-8-sig', newline='') as f:
                        f.write("# meta: параметры (первая строка — имена, вторая — значения)\n")
                        f.write(meta_df.to_csv(index=False))
                        f.write("\n# exp_epoch: средние значения показателей после 10 эпох\n")
                        f.write(summary.to_csv(index=False))

                messagebox.showinfo("Успех", f"Данные экспортированы в:\n{filename}")
            except Exception as e:
                messagebox.showerror("Ошибка", f"Ошибка экспорта: {str(e)}")
    
    def clear_frame(self):
        # Полное освобождение ресурсов matplotlib (предотвращение утечек)
        # 1. toolbar
        if hasattr(self, 'toolbar') and self.toolbar is not None:
            try:
                self.toolbar.destroy()
            except Exception:
                pass
            self.toolbar = None
        
        # 2. canvas
        if hasattr(self, 'canvas') and self.canvas is not None:
            try:
                self.canvas.get_tk_widget().destroy()
            except Exception:
                pass
            self.canvas = None
        
        # 3. figure
        if hasattr(self, 'fig') and self.fig is not None:
            try:
                self.fig.clf()
                # закрыть все axes
                for ax in self.fig.axes:
                    self.fig.delaxes(ax)
            except Exception:
                pass
            self.fig = None
        
        self.ax = None
        # двойная ось (ax.twinx) уничтожена вместе с fig — сброс, чтобы update_chart
        # не пытался удалить устаревшую ось (matplotlib KeyError)
        self._ax2 = None

        # 4. уничтожение Tkinter-виджетов
        for widget in self.main_frame.winfo_children():
            widget.destroy()
        
        if hasattr(self, 'param_entries'):
            self.param_entries.clear()
        
        self.root.unbind_all("<MouseWheel>")
        
        # 5. принудительный gc
        gc.collect()



if __name__ == '__main__':
    freeze_support()
    
    root = tk.Tk()
    app = TrafficModelGUI(root)
    root.mainloop()
