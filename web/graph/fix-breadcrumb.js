// Fix breadcrumb "Home" link to point to /wiki/ instead of /
(function() {
    // Find the breadcrumb Home link
    const homeLinks = document.querySelectorAll('.breadcrumb-element a[href="../"], .breadcrumb-element a[href="../../"], .breadcrumb-element a[href="../../../"]');
    
    homeLinks.forEach(link => {
        // Check if this is the "Home" link by checking its text content
        if (link.textContent.trim() === 'Home' || link.textContent.includes('Home')) {
            link.href = '/wiki/';
        }
    });
})();
