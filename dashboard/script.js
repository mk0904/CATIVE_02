document.addEventListener('DOMContentLoaded', () => {
    const navItems = document.querySelectorAll('.nav-links li');
    const sections = document.querySelectorAll('.view-section');

    navItems.forEach(item => {
        item.addEventListener('click', () => {
            // Remove active class from all nav items
            navItems.forEach(nav => nav.classList.remove('active'));
            
            // Add active class to clicked nav item
            item.classList.add('active');
            
            // Hide all sections
            sections.forEach(sec => sec.classList.remove('active'));
            
            // Show target section
            const targetId = item.getAttribute('data-target');
            document.getElementById(targetId).classList.add('active');
            
            // Trigger reflow to restart animations
            const targetSection = document.getElementById(targetId);
            targetSection.style.animation = 'none';
            targetSection.offsetHeight; /* trigger reflow */
            targetSection.style.animation = null; 
        });
    });

    // === INTERACTIVE EXPLORER LOGIC ===
    if (typeof startupData !== 'undefined') {
        const select = document.getElementById('startup-select');
        
        // Populate dropdown
        startupData.forEach((item, index) => {
            const option = document.createElement('option');
            option.value = index;
            option.textContent = item.name + ` | Funding: $${item.funding.toLocaleString()}`;
            select.appendChild(option);
        });

        // Helper for colors
        const getColor = (label) => {
            if (label === 'Emerging') return '#E07B54'; // Orange
            if (label === 'Growing') return '#5B8DB8';  // Blue
            return '#3FB950'; // Green (High Desirability)
        };

        // Handle selection
        select.addEventListener('change', (e) => {
            const data = startupData[e.target.value];
            
            // Update UI Raw Data
            document.getElementById('ui-sector').textContent = `${data.sector} | ${data.hq_city}`;
            document.getElementById('ui-roles').textContent = data.roles;
            document.getElementById('ui-funding').textContent = `$${data.funding.toLocaleString()} | ${data.employees} emp.`;
            document.getElementById('ui-ratings').textContent = `Cult: ${data.culture} | WLB: ${data.wlb} | Sal: ${data.salary}`;
            
            // Update Predictions
            const elTrue = document.getElementById('ui-true');
            elTrue.textContent = data.true_label;
            elTrue.style.color = getColor(data.true_label);
            
            const elSvm = document.getElementById('ui-svm');
            elSvm.textContent = data.svm_pred;
            elSvm.style.color = getColor(data.svm_pred);
            
            const elXgb = document.getElementById('ui-xgb');
            elXgb.textContent = data.xgb_pred;
            elXgb.style.color = getColor(data.xgb_pred);
            
            const elEnt = document.getElementById('ui-entropy');
            elEnt.textContent = `${data.entropy.toFixed(3)} (Max 1.09)`;
            elEnt.style.color = data.entropy > 0.8 ? '#E07B54' : '#3FB950';
        });

        // Trigger initial load
        select.dispatchEvent(new Event('change'));
    }
});
