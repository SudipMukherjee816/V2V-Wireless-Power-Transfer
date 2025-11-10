# v2v_streamlit_realistic.py
import streamlit as st
import pandas as pd
import random, uuid, math, hashlib
from dataclasses import dataclass, field
from typing import List, Dict
import altair as alt
import numpy as np

# --- Streamlit page config ---
st.set_page_config(page_title="V2V Inductive Charging — Realistic Demo", layout="wide")
st.title("Realistic V2V Dynamic Inductive Charging — Visual Dashboard")

st.markdown("""
**Features:**  
- Realistic dynamic inductive charging (2 m range, distance/velocity/lateral efficiency).  
- Dynamic trust evolution per transaction.  
- Altair charts for Energy/time, Latency, Trust, Vehicle positions.  
- CSV download buttons for logs and vehicle summary.
""")

# Sidebar controls
st.sidebar.header("Simulation Controls")
num_vehicles = st.sidebar.slider("Number of vehicles", 6, 40, 18)
rounds = st.sidebar.slider("Rounds", 5, 200, 40)
platoon_range = 3  # realistic inductive range in meters
edge_nodes = st.sidebar.checkbox("Use edge nodes (lower latency)", True)
seed = st.sidebar.number_input("Random seed", 0, 99999, 1234)
run_button = st.sidebar.button("Run Simulation")

# --- Vehicle class ---
@dataclass
class V:
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    battery_kwh: float = 50.0
    capacity_kwh: float = 75.0
    trust: float = 0.8
    role: str = "neutral"
    price_offer: float = 0.2
    max_transfer_rate: float = 5.0
    position_m: float = 0.0
    speed_kmh: float = 40.0
    marl_q: Dict[float,float] = field(default_factory=dict)
    price_grid: List[float] = field(default_factory=lambda:[0.08,0.12,0.16,0.20,0.25,0.30])
    eps: float = 0.15
    
    def decide_role(self):
        soc = self.battery_kwh / self.capacity_kwh
        if soc < 0.4:
            self.role = 'buyer'
        elif soc > 0.6:
            self.role = 'seller'
        else:
            self.role = 'neutral'

    def choose_price(self):
        for p in self.price_grid:
            if p not in self.marl_q:
                self.marl_q[p] = 0.0
        if random.random() < self.eps:
            self.price_offer = random.choice(self.price_grid)
        else:
            maxq = max(self.marl_q.values())
            cands = [p for p,q in self.marl_q.items() if q == maxq]
            self.price_offer = random.choice(cands)

# --- Helper functions ---
def align_prob(dx,lateral,rel_speed,max_range=platoon_range):
    """Return alignment probability (0–1) based on distance, lateral offset, relative speed"""
    p = 0.95
    if dx < 0.1 or dx > max_range:
        p *= 0.2
    else:
        p *= math.exp(-((dx - max_range/2)**2)/(2*(0.8**2)))  # Gaussian peak at mid-range
    p *= math.exp(-(lateral**2)/(2*(0.15**2)))
    p *= math.exp(-abs(rel_speed)/50.0)
    return max(0.0,min(1.0,p))

def latency_ms(edge): 
    base = 30 if edge else 80
    jitter = random.gauss(0,15)
    return max(5, base + jitter)

# --- Simulation function ---
def run_simulation(num_vehicles, rounds, edge_nodes, seed):
    random.seed(seed)
    np.random.seed(seed)
    
    road_len = 800.0
    vehicles = []
    for _ in range(num_vehicles):
        cap = random.choice([60.0,75.0,90.0])
        soc = random.uniform(0.12,0.96)
        v = V(
            battery_kwh=soc*cap,
            capacity_kwh=cap,
            trust=random.uniform(0.6,0.95),
            price_offer=random.choice([0.10,0.12,0.14,0.16,0.18,0.20,0.22,0.24,0.26]),
            max_transfer_rate=random.choice([3.0,5.0,7.0]),
            position_m=random.uniform(0,road_len),
            speed_kmh=random.uniform(20,80)
        )
        vehicles.append(v)
    
    ledger, logs = [], []
    
    for r in range(rounds):
        # Decide roles and prices
        for v in vehicles:
            v.decide_role()
            v.choose_price()
        
        buyers = [v for v in vehicles if v.role=='buyer']
        sellers = [v for v in vehicles if v.role=='seller']
        buyers_sorted = sorted(buyers, key=lambda x: x.price_offer, reverse=True)
        sellers_sorted = sorted(sellers, key=lambda x: x.price_offer)
        
        # Matching loop
        i=j=0
        while i<len(buyers_sorted) and j<len(sellers_sorted):
            b, s = buyers_sorted[i], sellers_sorted[j]
            if b.price_offer < s.price_offer: 
                i+=1
                continue
            
            dx = abs(b.position_m - s.position_m)
            if dx > platoon_range:  # realistic inductive range
                i+=1; j+=1; continue
            
            lateral = random.gauss(0,0.1)
            rel_speed = abs(b.speed_kmh - s.speed_kmh)
            p_align = align_prob(dx, lateral, rel_speed)
            
            lat = latency_ms(edge_nodes)
            price = (b.price_offer + s.price_offer)/2.0
            
            # Efficiency based on distance, speed, lateral
            efficiency = max(0, p_align * (1 - dx/platoon_range) * max(0,1 - rel_speed/50))
            # Allow minimum baseline transfer so weak alignments still exchange some energy
            transfer = max(0.1, efficiency * min(b.max_transfer_rate, s.max_transfer_rate, 
                                    max(0, 0.4*b.capacity_kwh - b.battery_kwh),
                                    max(0, s.battery_kwh - 0.6*s.capacity_kwh)))
            
            success = transfer > 0 and random.random() < min(1.0, 0.8 + (b.trust + s.trust)/3.0)
            
            if success:
                b.battery_kwh += transfer
                s.battery_kwh -= transfer
                amt = price * transfer
                tx = {'buyer':b.id,'seller':s.id,'energy_kwh':transfer,'price_per_kwh':price,
                      'amount':amt,'round':r,'latency_ms':lat,'dx_m':dx,'p_align':p_align}
                txh = hashlib.sha256(str(tx).encode()).hexdigest()[:12]
                tx['tx_hash'] = txh
                ledger.append(tx)
                logs.append({**tx,'success':True})
                
                # Update trust dynamically
                b.trust = min(1.0, b.trust + 0.02)
                s.trust = min(1.0, s.trust + 0.02)
            else:
                logs.append({'buyer':b.id,'seller':s.id,'energy_kwh':transfer,'price_per_kwh':price,
                             'amount':price*transfer,'round':r,'success':False,'dx_m':dx,
                             'p_align':p_align,'latency_ms':lat})
                
                b.trust = max(0.0, b.trust - 0.03)
                s.trust = max(0.0, s.trust - 0.03)
            
            i+=1; j+=1
            # --- Log unpaired buyers and sellers ---
        paired_buyers = {tx['buyer'] for tx in ledger}
        paired_sellers = {tx['seller'] for tx in ledger}

        for b in buyers_sorted:
            if b.id not in paired_buyers:
                logs.append({
                    'buyer': b.id,
                    'seller': None,
                    'energy_kwh': 0,
                    'price_per_kwh': b.price_offer,
                    'amount': 0,
                    'round': r,
                    'success': False,
                    'reason': 'no_seller_found'
                })

        for s in sellers_sorted:
            if s.id not in paired_sellers:
                logs.append({
                    'buyer': None,
                    'seller': s.id,
                    'energy_kwh': 0,
                    'price_per_kwh': s.price_offer,
                    'amount': 0,
                    'round': r,
                    'success': False,
                    'reason': 'no_buyer_found'
                })

        
        # Move vehicles
        for v in vehicles:
            dt = 0.01
            v.position_m = (v.position_m + v.speed_kmh*1000/3600*dt) % road_len
    
    df_logs = pd.DataFrame(logs)
    df_veh = pd.DataFrame([{'id':v.id,'soc':v.battery_kwh/v.capacity_kwh,'battery_kwh':v.battery_kwh,
                            'capacity_kwh':v.capacity_kwh,'trust':v.trust,'price_offer':v.price_offer,
                            'pos_m':v.position_m,'speed_kmh':v.speed_kmh} for v in vehicles])
    df_ledger = pd.DataFrame(ledger)
    return df_logs, df_veh, df_ledger

# --- Run simulation ---
if run_button:
    with st.spinner("Running simulation..."):
        df_logs, df_veh, df_ledger = run_simulation(num_vehicles, rounds, edge_nodes, seed)
    st.success("Simulation complete")
    
    # Energy per round
    if 'success' in df_logs.columns and not df_logs.empty:
        df_energy = df_logs[df_logs['success']==True].groupby('round', as_index=False)['energy_kwh'].sum()
    else:
        df_energy = pd.DataFrame({'round':[], 'energy_kwh':[]})

    line = alt.Chart(df_energy).mark_line(point=True).encode(
        x=alt.X('round:Q', title='Round'),
        y=alt.Y('energy_kwh:Q', title='Energy (kWh)'),
        tooltip=['round','energy_kwh']
    ).properties(height=300, width=600, title="Energy transferred per round")
    st.altair_chart(line, use_container_width=True)
    
    # Latency histogram
    if 'latency_ms' in df_logs.columns and not df_logs['latency_ms'].dropna().empty:
        hist_lat = alt.Chart(df_logs).mark_bar().encode(
            alt.X('latency_ms:Q', bin=alt.Bin(maxbins=30), title='Latency (ms)'),
            y='count()'
        ).properties(height=200, width=450, title="Latency distribution")
        st.altair_chart(hist_lat, use_container_width=False)
    
    # Trust distribution
    hist_trust = alt.Chart(df_veh).mark_bar().encode(
        alt.X('trust:Q', bin=alt.Bin(maxbins=20), title='Trust'),
        y='count()'
    ).properties(height=200, width=450, title="Dynamic trust distribution")
    st.altair_chart(hist_trust, use_container_width=False)
    
    # Vehicle positions (1D)
    df_veh['jitter'] = np.random.uniform(0, 10, len(df_veh))
    pos_chart = alt.Chart(df_veh).mark_circle(size=80).encode(
        x=alt.X('pos_m:Q', title='Position along road (m)'),
        y=alt.Y('jitter:Q', title='', axis=None),
        color=alt.Color('trust:Q', scale=alt.Scale(scheme='blues'), title='Trust'),
        tooltip=['id','soc','trust','speed_kmh']
    ).properties(height=150, width=800, title="Vehicle positions (1D map)")
    st.altair_chart(pos_chart, use_container_width=True)
    
    # Show logs and ledger
    st.subheader("Transactions Log")
    st.write(df_logs)
    if not df_ledger.empty:
        st.subheader("Ledger (recorded transactions)")
        st.write(df_ledger)
        st.download_button("Download ledger CSV", data=df_ledger.to_csv(index=False), file_name="v2v_ledger.csv", mime="text/csv")
    
else:
    st.info("Set parameters in the sidebar and click **Run Simulation** to generate interactive charts and downloadable CSVs.")
