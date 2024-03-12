import React, { useEffect, useState } from 'react';

function App() {
    const [htmlContent, setHtmlContent] = useState('');

    useEffect(() => {
        fetch('/frontend/project_frontend.html') // Update path to your HTML file
            .then(response => response.text())
            .then(html => {
                setHtmlContent(html);
            })
            .catch(error => console.error('Error fetching notebook content:', error));
    }, []);

    return (
        <div dangerouslySetInnerHTML={{ __html: htmlContent }} />
    );
}

export default App;