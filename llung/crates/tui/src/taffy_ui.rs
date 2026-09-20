use ratatui::{
    layout::Rect,
    style::{Color, Style},
};
use taffy::{
    AvailableSpace, Dimension, Display, FlexDirection, NodeId, Size, Style as TaffyStyle, TaffyTree,
};

pub enum NodeContent {
    /// Just a background color fill (e.g., sidebar panel)
    Block { bg: Color },
    /// Wrapped text rendered with prefix sums
    Text {
        content: crate::prefix_text::PrefixText,
        style: Style,
    },
    /// Bordered box, optionally titled
    Container { title: Option<String>, border: bool },
}

pub struct TaffyUi {
    taffy: TaffyTree,
    root: NodeId,
    pub sidebar: NodeId,
    pub chat: NodeId,
    pub input: NodeId,
}

impl TaffyUi {
    /// Build a simple chat layout: sidebar | main area
    pub fn new_chat_layout(term_w: u16, term_h: u16, sidebar_visible: bool) -> Self {
        let mut taffy = TaffyTree::new();

        // Sidebar: fixed 30 columns, full height
        let sidebar = taffy
            .new_leaf(TaffyStyle {
                size: Size {
                    width: if sidebar_visible {
                        Dimension::length(30.0)
                    } else {
                        Dimension::length(0.0)
                    },
                    height: Dimension::percent(1.0),
                },
                ..Default::default()
            })
            .unwrap();

        let chat = taffy
            .new_leaf(TaffyStyle {
                size: Size {
                    width: Dimension::percent(1.0),
                    height: Dimension::auto(),
                },
                flex_grow: 1.0,
                ..Default::default()
            })
            .unwrap();

        let input = taffy
            .new_leaf(TaffyStyle {
                size: Size {
                    width: Dimension::percent(1.0),
                    height: Dimension::length(3.0),
                },
                ..Default::default()
            })
            .unwrap();

        // Main area: fills remaining space
        let main_col = taffy
            .new_with_children(
                TaffyStyle {
                    display: Display::Flex,
                    flex_direction: FlexDirection::Column,
                    size: Size {
                        width: Dimension::auto(),
                        height: Dimension::percent(1.0),
                    },
                    flex_grow: 1.0,
                    ..Default::default()
                },
                &[chat, input],
            )
            .unwrap();

        // Root: horizontal flex row
        let root = taffy
            .new_with_children(
                TaffyStyle {
                    size: Size {
                        width: Dimension::length(term_w as f32),
                        height: Dimension::length(term_h as f32),
                    },
                    display: Display::Flex,
                    flex_direction: FlexDirection::Row,
                    ..Default::default()
                },
                &[sidebar, main_col],
            )
            .unwrap();

        Self {
            taffy,
            root,
            sidebar,
            chat,
            input,
        }
    }

    /// Recompute layout when the terminal resizes.
    pub fn resize(&mut self, w: u16, h: u16) {
        let _ = self.taffy.compute_layout(
            self.root,
            Size {
                width: AvailableSpace::Definite(w as f32),
                height: AvailableSpace::Definite(h as f32),
            },
        );
    }

    pub fn node_rect(&self, node: NodeId) -> Rect {
        let mut x = 0.0;
        let mut y = 0.0;
        let mut current = Some(node);

        while let Some(n) = current {
            let layout = self.taffy.layout(n).unwrap();
            x += layout.location.x;
            y += layout.location.y;
            current = self.taffy.parent(n);
        }
        let layout = self.taffy.layout(node).unwrap();
        Rect::new(
            x as u16,
            y as u16,
            layout.size.width as u16,
            layout.size.height as u16,
        )
    }
    pub fn set_sidebar_visible(&mut self, visible: bool) {
        let style = taffy::Style {
            size: Size {
                width: if visible {
                    Dimension::length(30.0)
                } else {
                    Dimension::length(0.0)
                },
                height: Dimension::percent(1.0),
            },
            ..Default::default()
        };
        let _ = self.taffy.set_style(self.sidebar, style);
    }
}
