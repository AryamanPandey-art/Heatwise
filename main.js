document.addEventListener('DOMContentLoaded', () => {
    // 1. Configurable Application URL Routing
    const getAppUrl = () => window.HEATWISE_APP_URL || 'http://localhost:8501';
    const syncAppUrls = () => {
        const url = getAppUrl();
        document.querySelectorAll('a[data-app-cta], .launch-app-btn, a[href*="localhost:8501"]').forEach(el => {
            el.href = url;
            el.target = '_blank';
            el.rel = 'noopener noreferrer';
        });
    };
    syncAppUrls();

    document.addEventListener('click', (e) => {
        const cta = e.target.closest('a[data-app-cta], .launch-app-btn, a[href*="localhost:8501"]');
        if (cta) {
            cta.href = getAppUrl();
        }
    });

    // 2. Theme Toggle (Sync between floating button and navbar button)
    const htmlEl = document.documentElement;
    const themeBtnFloating = document.getElementById('themeToggle');
    const themeBtnNav = document.getElementById('navThemeToggle');

    const updateThemeUI = (theme) => {
        if (theme === 'light') {
            htmlEl.classList.remove('dark');
            htmlEl.classList.add('light');
            if (themeBtnNav) {
                themeBtnNav.querySelector('.theme-text').innerText = 'Dark';
            }
        } else {
            htmlEl.classList.remove('light');
            htmlEl.classList.add('dark');
            if (themeBtnNav) {
                themeBtnNav.querySelector('.theme-text').innerText = 'Light';
            }
        }
    };

    // Check URL param first (?theme=light or ?theme=dark), then saved preference, default to dark
    const urlParams = new URLSearchParams(window.location.search);
    const themeParam = urlParams.get('theme');
    const savedTheme = themeParam || localStorage.getItem('theme') || 'dark';
    updateThemeUI(savedTheme);

    const toggleTheme = () => {
        const isDark = htmlEl.classList.contains('dark');
        const nextTheme = isDark ? 'light' : 'dark';
        updateThemeUI(nextTheme);
        localStorage.setItem('theme', nextTheme);
    };

    if (themeBtnFloating) themeBtnFloating.addEventListener('click', toggleTheme);
    if (themeBtnNav) themeBtnNav.addEventListener('click', toggleTheme);

    // 3. Navbar scroll effect & Active Nav Spy
    const navbar = document.getElementById('navbar');
    const navItems = document.querySelectorAll('.nav-item');
    const sections = document.querySelectorAll('section[id]');

    window.addEventListener('scroll', () => {
        if (window.scrollY > 30) {
            navbar.style.boxShadow = htmlEl.classList.contains('light') 
                ? '0 4px 20px rgba(0,0,0,0.06)' 
                : '0 4px 25px rgba(0,0,0,0.5)';
        } else {
            navbar.style.boxShadow = 'none';
        }

        // Active link spy
        let currentSection = 'home';
        sections.forEach(sec => {
            const top = sec.offsetTop - 120;
            if (window.scrollY >= top) {
                currentSection = sec.getAttribute('id');
            }
        });

        navItems.forEach(item => {
            item.classList.remove('active');
            if (item.getAttribute('href') === `#${currentSection}`) {
                item.classList.add('active');
            }
        });
    });

    // 4. Interactive Simulation Demo Logic
    const grid = document.getElementById('demoRacksGrid');
    const cList = document.getElementById('candidateList');
    const btn = document.getElementById('runSimulationBtn');
    const decisionBox = document.querySelector('.decision-outcome-box');
    const decisionTarget = document.getElementById('decisionTarget');
    const decisionReason = document.getElementById('decisionReason');

    // 8 Realistic Rack initial states
    const racksData = [
        { id: 1, name: 'RACK 01', temp: 58.2, maxTemp: 85 },
        { id: 2, name: 'RACK 02', temp: 71.4, maxTemp: 85 },
        { id: 3, name: 'RACK 03', temp: 46.1, maxTemp: 85 },
        { id: 4, name: 'RACK 04', temp: 82.5, maxTemp: 85 },
        { id: 5, name: 'RACK 05', temp: 52.8, maxTemp: 85 },
        { id: 6, name: 'RACK 06', temp: 64.9, maxTemp: 85 },
        { id: 7, name: 'RACK 07', temp: 44.3, maxTemp: 85 },
        { id: 8, name: 'RACK 08', temp: 76.0, maxTemp: 85 },
    ];

    const getTempColor = (t) => {
        if (t < 55) return '#10b981'; // Green
        if (t < 75) return '#f59e0b'; // Amber
        return '#ef4444'; // Red
    };

    const renderRacks = () => {
        if (!grid) return;
        grid.innerHTML = '';
        racksData.forEach(r => {
            const card = document.createElement('div');
            card.className = 'rack-card-unit';
            card.id = `rack-node-${r.id}`;

            const color = getTempColor(r.temp);
            const fillPct = Math.min(100, (r.temp / 90) * 100);

            card.innerHTML = `
                <div class="rack-card-top">
                    <span>${r.name}</span>
                    <span style="color: ${color}">●</span>
                </div>
                <div class="rack-temp-tag" style="color: ${color}">
                    ${r.temp.toFixed(1)}°C
                </div>
                <div class="rack-heat-bar-track">
                    <div class="rack-heat-bar-fill" style="width: ${fillPct}%; background-color: ${color};"></div>
                </div>
            `;
            grid.appendChild(card);
        });
    };

    renderRacks();

    if (btn) {
        btn.addEventListener('click', () => {
            renderRacks();
            if (cList) cList.innerHTML = '';
            if (decisionBox) decisionBox.style.opacity = '0.3';
            btn.disabled = true;
            btn.innerText = "Evaluating Thermal Physics...";

            let index = 0;
            const evaluations = [];

            const evalInterval = setInterval(() => {
                if (index >= racksData.length) {
                    clearInterval(evalInterval);
                    finishPlacement(evaluations);
                    return;
                }

                const currentRack = racksData[index];
                const rackEl = document.getElementById(`rack-node-${currentRack.id}`);
                if (rackEl) rackEl.classList.add('evaluating');

                // Physics simulation prediction: baseTemp + workload induced delta
                const wlType = document.getElementById('wlType').value;
                const heatFactor = wlType === 'AI Training' ? 7.2 : (wlType === 'LLM Inference' ? 4.8 : 2.1);
                const predictedTemp = currentRack.temp + heatFactor + (Math.random() * 1.5 - 0.7);

                evaluations.push({
                    id: currentRack.id,
                    name: currentRack.name,
                    predicted: predictedTemp
                });

                if (cList) {
                    const li = document.createElement('li');
                    li.className = 'cand-item';
                    const col = getTempColor(predictedTemp);
                    li.innerHTML = `
                        <span class="cand-rack-name">${currentRack.name}</span>
                        <span class="cand-temp-val" style="color: ${col}">Est: ${predictedTemp.toFixed(1)}°C</span>
                    `;
                    cList.appendChild(li);
                    cList.scrollTop = cList.scrollHeight;
                }

                setTimeout(() => {
                    if (rackEl) rackEl.classList.remove('evaluating');
                }, 220);

                index++;
            }, 140);
        });
    }

    const finishPlacement = (evaluations) => {
        btn.disabled = false;
        btn.innerText = "Simulate Workload Placement";

        // Sort by predicted lowest temperature
        evaluations.sort((a, b) => a.predicted - b.predicted);
        const optimal = evaluations[0];

        // Highlight selected rack
        const winnerEl = document.getElementById(`rack-node-${optimal.id}`);
        if (winnerEl) {
            winnerEl.classList.add('selected');
        }

        // Apply temperature rise to winner rack
        const rackObj = racksData.find(r => r.id === optimal.id);
        if (rackObj) {
            rackObj.temp = optimal.predicted;
        }

        if (decisionTarget) decisionTarget.innerText = optimal.name;
        if (decisionReason) {
            decisionReason.innerText = `"Optimally placed in ${optimal.name}: Lowest post-placement thermal rise (${optimal.predicted.toFixed(1)}°C), maintaining 22°C safe thermal headroom."`;
        }
        if (decisionBox) decisionBox.style.opacity = '1';

        setTimeout(() => renderRacks(), 800);
    };

    // 5. Workload select stats synchronization
    const wlTypeSelect = document.getElementById('wlType');
    const wlGpu = document.getElementById('wlGpu');
    const wlMem = document.getElementById('wlMem');
    const wlPwr = document.getElementById('wlPwr');

    if (wlTypeSelect) {
        wlTypeSelect.addEventListener('change', (e) => {
            const val = e.target.value;
            if (val === 'AI Training') {
                if (wlGpu) wlGpu.innerText = '80%';
                if (wlMem) wlMem.innerText = '70%';
                if (wlPwr) wlPwr.innerText = '2.0 kW';
            } else if (val === 'LLM Inference') {
                if (wlGpu) wlGpu.innerText = '60%';
                if (wlMem) wlMem.innerText = '85%';
                if (wlPwr) wlPwr.innerText = '1.5 kW';
            } else {
                if (wlGpu) wlGpu.innerText = '25%';
                if (wlMem) wlMem.innerText = '30%';
                if (wlPwr) wlPwr.innerText = '0.5 kW';
            }
        });
    }
});
