document.addEventListener('DOMContentLoaded', () => {
    const homeBtn = document.querySelector('.nav-link[href="index.html"]');
    const aboutBtn = document.querySelector('.dropdown-button');
    const logsBtn = document.querySelector('.nav-link[href="logs.html"]');
    const dropdownContent = document.querySelector('.dropdown-content');

    homeBtn.addEventListener('click', (event) => {
        event.preventDefault();
        window.location.href = 'index.html';
    });

    aboutBtn.addEventListener('click', () => {
        const dropdownVisible = dropdownContent.style.display === 'block';
        dropdownContent.style.display = dropdownVisible ? 'none' : 'block';
    });

    logsBtn.addEventListener('click', (event) => {
        event.preventDefault();
        window.location.href = 'logs.html';
    });

    const currentPage = window.location.pathname.split('/').pop();
    if (currentPage === 'about.html' || currentPage === 'purpose.html' || currentPage === 'ai_models.html') {
        aboutBtn.classList.add('stretched'); 
    }
});
